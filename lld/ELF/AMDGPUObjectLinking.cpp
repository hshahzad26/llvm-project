//===- AMDGPUObjectLinking.cpp - AMDGPU link-time resolution --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements link-time resolution and patching for AMDGPU object linking.
//
// The linker:
//   1. Validates target-ID compatibility for participating objects
//   2. Collects SHN_AMDGPU_LDS and named-barrier symbols
//   3. Parses .amdgpu.info to build the cross-TU call graph, type-ID
//      signatures, LDS and named-barrier uses, and per-function resource usage
//   4. Resolves indirect call edges, function aliases, and kernel entries
//   5. Validates call-edge wave-size compatibility
//   6. Builds a shared SCC condensation graph for kernel-reachable functions
//   7. Computes per-kernel LDS and named-barrier reachability
//   8. Assigns LDS offsets and named-barrier IDs
//   9. Propagates resource usage across the SCC graph (MAX for registers, OR
//      for flags, caller scratch plus maximum callee scratch path)
//  10. Validates required ABI occupancy metadata, call-edge occupancy
//      compatibility, and each kernel's LDS usage against its occupancy
//
// After resolution, the linker patches kernel descriptors and HSA metadata
// with the resolved LDS size, named-barrier count, propagated register usage,
// scratch size, dynamic-stack flag, and related resource fields.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUObjectLinking.h"
#include "Config.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/BinaryFormat/AMDGPUMetadataVerifier.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/AMDGPUIsaInfo.h"
#include "llvm/Support/AMDGPUObjLinkingInfo.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/TargetParser/Triple.h"

#define DEBUG_TYPE "amdgpu-object-linking"

using namespace llvm;
using namespace llvm::support::endian;
using namespace llvm::object;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

namespace llvm::AMDGPU {
StringRef getArchNameFromElfMach(unsigned elfMach);
} // namespace llvm::AMDGPU

namespace {

// One SHN_AMDGPU_LDS symbol after collection and before it is rewritten to a
// concrete offset.
struct LDSSymbolInfo {
  Symbol *sym;
  uint64_t size;
  Align alignment;
  uint64_t assignedOffset = 0;
};

// One named-barrier pseudo-symbol. The linker assigns each barrier a hardware
// ID range and rewrites the symbol to the encoded barrier address.
struct NamedBarrierInfo {
  Symbol *sym;
  uint32_t slotCount;
  uint32_t assignedBarId = 0;
};

// Resource usage for one function or SCC. Used both for direct records from
// .amdgpu.info and for propagated (transitive) results. The occupancy field is
// only meaningful for direct (local) records.
struct ResourceInfo {
  uint32_t numArchVGPR = 0;
  uint32_t numAccVGPR = 0;
  uint32_t numSGPR = 0;
  uint32_t scratchSize = 0;
  uint32_t occupancy = 0;
  uint32_t waveSize = 0;
  bool usesVCC = false;
  bool usesFlatScratch = false;
  bool hasDynSizedStack = false;
};

// One function in the device call graph, plus the data computed for that
// function when it is a kernel entry.
struct CGNode {
  Symbol *sym = nullptr;
  bool isKernel = false;

  SmallVector<CGNode *, 4> callees;
  SmallVector<size_t, 2> ldsUseIndices;
  SmallVector<size_t, 2> barrierUseIndices;

  ResourceInfo localRes;
  bool hasLocalRes = false;

  ResourceInfo propagatedRes;
  bool hasPropagatedRes = false;

  DenseSet<size_t> reachableLDS;
  DenseSet<size_t> reachableBarriers;
  uint32_t ldsSize = 0;
  uint32_t numNamedBarrier = 0;
};

// Functions and indirect callers that share the same type-id encoding.
struct SignatureInfo {
  SmallVector<CGNode *, 4> functions;
  SmallVector<CGNode *, 4> indirectCallers;
};

// One strongly connected component in the kernel-reachable call graph. The SCC
// carries direct and propagated reachability/resource data so LDS,
// named-barrier, and resource passes can share the same condensed graph.
struct CallGraphSCC {
  SmallVector<CGNode *, 4> members;
  SmallVector<unsigned, 4> callees;

  DenseSet<size_t> reachableLDS;
  DenseSet<size_t> reachableBarriers;

  ResourceInfo localRes;
  ResourceInfo propagatedRes;

  bool hasPropagatedRes = false;
  bool isRecursive = false;
};

// Cross-object device call graph built from .amdgpu.info records. The graph
// keeps kernel order stable for diagnostics and later per-kernel patching.
// After construction, the graph can be condensed into an SCC DAG for
// reachability and resource propagation.
class AMDGPUCallGraph {
  SpecificBumpPtrAllocator<CGNode> alloc;
  DenseMap<Symbol *, CGNode *> symToNode;
  SmallVector<CGNode *> kernelNodes;
  bool hasLDSUseEntries = false;
  bool hasBarrierUseEntries = false;

  DenseSet<CGNode *> addressTakenNodes;
  StringMap<SignatureInfo> signatureMap;

  // SCC condensation state (populated by buildCondensedGraph).
  SmallVector<CallGraphSCC, 16> sccs;
  DenseMap<CGNode *, unsigned> nodeToSCC;

public:
  CGNode &getOrCreate(Symbol *sym) {
    auto [it, inserted] = symToNode.try_emplace(sym, nullptr);
    if (inserted) {
      it->second = new (alloc.Allocate()) CGNode();
      it->second->sym = sym;
    }
    return *it->second;
  }

  CGNode *lookup(Symbol *sym) const {
    auto it = symToNode.find(sym);
    return it != symToNode.end() ? it->second : nullptr;
  }

  void addKernel(CGNode &node) {
    if (node.isKernel)
      return;
    node.isKernel = true;
    kernelNodes.push_back(&node);
  }

  void markAddressTaken(CGNode *node) { addressTakenNodes.insert(node); }

  void addIndirectCall(CGNode *caller, StringRef encoding) {
    signatureMap[encoding].indirectCallers.push_back(caller);
  }

  void addSignature(CGNode *node, StringRef encoding) {
    signatureMap[encoding].functions.push_back(node);
  }

  // After all sections are parsed, resolve indirect call edges by matching
  // signature encodings: for each indirect call encoding, the potential callees
  // are address-taken functions with the same encoding.
  void buildIndirectEdges() {
    for (auto &[encoding, info] : signatureMap) {
      if (info.indirectCallers.empty())
        continue;

      SmallVector<CGNode *, 4> potentialCallees;
      for (CGNode *func : info.functions) {
        if (addressTakenNodes.count(func))
          potentialCallees.push_back(func);
      }

      if (potentialCallees.empty())
        continue;

      for (CGNode *caller : info.indirectCallers) {
        for (CGNode *callee : potentialCallees) {
          LLVM_DEBUG(dbgs() << "  indirect edge: " << caller->sym->getName()
                            << " -> " << callee->sym->getName()
                            << " (sig=" << encoding << ")\n");
          caller->callees.push_back(callee);
        }
      }
    }
  }

  // Resolve function aliases: multiple ELF symbols can point to the same
  // address (e.g. weak/strong pairs or constructor variants). The compiler
  // emits .amdgpu.info only for one of them. Redirect callee pointers from
  // alias nodes (no local resource info) to their canonical definition so
  // that reachability and resource propagation see real data.
  void resolveAliases() {
    DenseMap<std::pair<SectionBase *, uint64_t>, CGNode *> addrToNode;
    for (auto &[sym, node] : symToNode) {
      if (!node->hasLocalRes)
        continue;
      auto *d = dyn_cast<Defined>(sym);
      if (!d || !d->section)
        continue;
      addrToNode[{d->section, d->value}] = node;
    }

    DenseMap<CGNode *, CGNode *> aliasMap;
    for (auto &[sym, node] : symToNode) {
      if (node->hasLocalRes)
        continue;
      auto *d = dyn_cast<Defined>(sym);
      if (!d || !d->section)
        continue;
      auto it = addrToNode.find({d->section, d->value});
      if (it != addrToNode.end() && it->second != node) {
        LLVM_DEBUG(dbgs() << "  alias: " << sym->getName() << " -> "
                          << it->second->sym->getName() << "\n");
        aliasMap[node] = it->second;
      }
    }

    if (aliasMap.empty())
      return;

    for (auto &[sym, node] : symToNode) {
      for (CGNode *&callee : node->callees) {
        if (auto it = aliasMap.find(callee); it != aliasMap.end())
          callee = it->second;
      }
    }

    for (CGNode *&k : kernelNodes) {
      if (auto it = aliasMap.find(k); it != aliasMap.end())
        k = it->second;
    }
  }

  void setHasLDSUses() { hasLDSUseEntries = true; }
  bool hasLDSUses() const { return hasLDSUseEntries; }

  void setHasBarrierUses() { hasBarrierUseEntries = true; }
  bool hasBarrierUses() const { return hasBarrierUseEntries; }

  ArrayRef<CGNode *> kernels() const { return kernelNodes; }

  using const_iterator = DenseMap<Symbol *, CGNode *>::const_iterator;
  const_iterator begin() const { return symToNode.begin(); }
  const_iterator end() const { return symToNode.end(); }

  // Build the kernel-reachable SCC DAG. Tarjan emits SCCs in reverse
  // topological order so every outgoing edge points to an earlier SCC and
  // consumers can use a single forward sweep.
  bool buildCondensedGraph(Ctx &ctx, bool validateResources);

  // Propagate transitive LDS/barrier reachability over the SCC DAG into each
  // kernel node.
  void computeKernelReachability();

  // Propagate resource usage over the SCC DAG. Since SCCs are in reverse
  // topological order, each callee has already been propagated when its
  // caller is visited.
  void propagateResourceUsage();

private:
  bool buildSCCs(Ctx &ctx, CGNode *node,
                 DenseMap<CGNode *, unsigned> &nodeIndex,
                 DenseMap<CGNode *, unsigned> &lowLink,
                 DenseSet<CGNode *> &onStack, SmallVectorImpl<CGNode *> &stack,
                 unsigned &nextIndex, bool validateResources);

  void buildSCCEdges();
};

} // namespace

//===----------------------------------------------------------------------===//
// LDS symbol collection
//===----------------------------------------------------------------------===//

// Collect every distinct SHN_AMDGPU_LDS common symbol that needs a final
// link-time offset. Named barrier symbols (identified by the
// __amdgpu_named_barrier prefix) are routed to a separate barriers array.
static void collectLDSSymbols(Ctx &ctx,
                              SmallVectorImpl<LDSSymbolInfo> &ldsSymbols,
                              SmallVectorImpl<NamedBarrierInfo> &barriers) {
  DenseSet<Symbol *> seen;
  for (ELFFileBase *file : ctx.objectFiles) {
    if (!file->hasCommonSyms)
      continue;
    for (Symbol *sym : file->getGlobalSymbols()) {
      if (!sym->isAMDGPULDS || !seen.insert(sym).second)
        continue;
      auto *cs = dyn_cast<CommonSymbol>(sym);
      if (!cs)
        continue;
      if (sym->getName().starts_with("__amdgpu_named_barrier")) {
        LLVM_DEBUG(dbgs() << "  collected named barrier: " << sym->getName()
                          << " size=" << cs->size << "\n");
        barriers.push_back({sym, static_cast<uint32_t>(cs->size / 16)});
      } else {
        LLVM_DEBUG(dbgs() << "  collected LDS symbol: " << sym->getName()
                          << " size=" << cs->size << " align=" << cs->alignment
                          << "\n");
        ldsSymbols.push_back(
            {sym, cs->size, Align(cs->alignment), /*assignedOffset=*/0});
      }
    }
  }
}

// Sort larger and more-aligned LDS objects first to reduce padding while
// keeping symbol-name ordering deterministic for ties.
static bool compareLDSSymbol(const LDSSymbolInfo &a, const LDSSymbolInfo &b) {
  if (a.alignment != b.alignment)
    return a.alignment > b.alignment;
  if (a.size != b.size)
    return a.size > b.size;
  return a.sym->getName() < b.sym->getName();
}

// Assign a single packed layout used by all kernels. This is the fallback when
// no .amdgpu.info reachability data is available.
static void assignUniversalOffsets(SmallVectorImpl<LDSSymbolInfo> &ldsSymbols) {
  llvm::sort(ldsSymbols, compareLDSSymbol);
  uint64_t currentOffset = 0;
  for (LDSSymbolInfo &lds : ldsSymbols) {
    if (lds.size == 0)
      continue;
    currentOffset = alignTo(currentOffset, lds.alignment);
    lds.assignedOffset = currentOffset;
    currentOffset += lds.size;
  }
  Align maxDynAlign(1);
  for (LDSSymbolInfo &lds : ldsSymbols) {
    if (lds.size == 0)
      maxDynAlign = std::max(maxDynAlign, Align(lds.alignment));
  }
  uint64_t dynBase = alignTo(currentOffset, maxDynAlign);
  for (LDSSymbolInfo &lds : ldsSymbols) {
    if (lds.size == 0)
      lds.assignedOffset = dynBase;
  }
}

//===----------------------------------------------------------------------===//
// Section parsing
//===----------------------------------------------------------------------===//

// Build relocation map: byte offset -> {ELF symbol index, addend}.
// The addend is needed to resolve STT_SECTION symbols: ELF assemblers
// canonically convert relocations for local symbols in their own sections
// into section_symbol + addend form, so we must track the addend to map
// back to the actual function symbol (see resolveSecSymbol).
namespace {
// The relocation payload needed to resolve a tagged .amdgpu.info entry.
struct RelocInfo {
  uint32_t symIdx;
  int64_t addend;
};
} // namespace

// Extract explicit RELA/CREL addends and treat REL entries as addend zero.
template <class RelTy> static int64_t getAddend(const RelTy &rel) {
  if constexpr (RelTy::HasAddend)
    return rel.r_addend;
  return 0;
}

// Build the relocation lookup for one .amdgpu.info section.
template <class ELFT>
static DenseMap<uint64_t, RelocInfo> buildRelocMap(ObjFile<ELFT> *obj,
                                                   uint32_t secIndex) {
  ArrayRef<typename ELFT::Shdr> objSections = obj->template getELFShdrs<ELFT>();
  const ELFFile<ELFT> &elfObj = obj->getObj();
  DenseMap<uint64_t, RelocInfo> relocMap;
  for (size_t i = 0, e = objSections.size(); i < e; ++i) {
    const auto &relSec = objSections[i];
    if (relSec.sh_info != secIndex)
      continue;
    if (relSec.sh_type == SHT_RELA) {
      for (const auto &rel :
           CHECK(elfObj.relas(relSec), "could not read rela section"))
        relocMap[rel.r_offset] = {rel.getSymbol(false), getAddend(rel)};
      break;
    }
    if (relSec.sh_type == SHT_REL) {
      for (const auto &rel :
           CHECK(elfObj.rels(relSec), "could not read rel section"))
        relocMap[rel.r_offset] = {rel.getSymbol(false), 0};
      break;
    }
    if (relSec.sh_type == SHT_CREL) {
      auto crels = CHECK(elfObj.crels(relSec), "could not read crel section");
      for (const auto &rel : crels.first)
        relocMap[rel.r_offset] = {rel.getSymbol(false), getAddend(rel)};
      for (const auto &rel : crels.second)
        relocMap[rel.r_offset] = {rel.getSymbol(false), getAddend(rel)};
      break;
    }
  }
  return relocMap;
}

// Map function definition addresses back to symbols so section-symbol
// relocations can recover the original named function.
template <class ELFT>
static DenseMap<std::pair<SectionBase *, uint64_t>, Symbol *>
buildSymbolAddressMap(ObjFile<ELFT> *obj) {
  DenseMap<std::pair<SectionBase *, uint64_t>, Symbol *> addrToSym;
  for (Symbol *sym : obj->getSymbols()) {
    if (!sym || sym->getName().empty())
      continue;
    auto *def = dyn_cast<Defined>(sym);
    if (!def || !def->section)
      continue;
    addrToSym.try_emplace({def->section, def->value}, sym);
  }
  return addrToSym;
}

// Resolve an STT_SECTION symbol + addend to the named function symbol at that
// address. ELF assemblers replace relocations targeting local symbols with
// section_symbol + addend (standard ELF canonicalization). Since section
// symbols have empty names in LLD, the .amdgpu.info parser cannot identify
// the function from the section symbol alone. Use a per-object address map to
// find the named Defined symbol at the matching section and offset.
static Symbol *resolveSecSymbol(
    Symbol &secSym, int64_t addend,
    const DenseMap<std::pair<SectionBase *, uint64_t>, Symbol *> &addrToSym) {
  auto *secDef = dyn_cast<Defined>(&secSym);
  if (!secDef || !secDef->section)
    return nullptr;
  auto it = addrToSym.find({secDef->section, static_cast<uint64_t>(addend)});
  return it == addrToSym.end() ? nullptr : it->second;
}

using namespace llvm::AMDGPU;

// Resolve a relocation attached to a .amdgpu.info payload field. The result is
// either the referenced symbol or the named function represented by a
// section-symbol relocation.
template <class ELFT>
static Symbol *resolveRelocSym(
    ObjFile<ELFT> *obj, const DenseMap<uint64_t, RelocInfo> &relocMap,
    const DenseMap<std::pair<SectionBase *, uint64_t>, Symbol *> &addrToSym,
    uint64_t off, StringRef diagName) {
  auto it = relocMap.find(off);
  if (it == relocMap.end())
    return nullptr;
  Symbol *sym = &obj->getSymbol(it->second.symIdx);
  if (sym->getName().empty()) {
    sym = resolveSecSymbol(*sym, it->second.addend, addrToSym);
    if (!sym) {
      Warn(obj->ctx) << obj->getName() << ": .amdgpu.info " << diagName
                     << " at offset " << off
                     << " references a section symbol that could not be "
                        "resolved to a named function symbol";
    }
  }
  return sym;
}

// Return a string from .amdgpu.strtab without reading past the section if
// malformed input omits the trailing NUL.
static StringRef getInfoString(StringRef strtab, uint32_t offset) {
  if (offset >= strtab.size())
    return StringRef();
  StringRef tail = strtab.substr(offset);
  size_t nul = tail.find('\0');
  return tail.substr(0, nul);
}

// .amdgpu.info is emitted as a standalone section, so records from losing weak
// definitions or discarded COMDAT groups can survive. Only the prevailing
// definition may contribute local resources and edges. Non-prevailing scopes
// may still contribute type IDs for address-taken declarations.
static bool isPrevailingInfoFunction(ELFFileBase *obj, Symbol *sym) {
  auto *def = dyn_cast<Defined>(sym);
  if (!def || !def->section || def->section == &InputSection::discarded)
    return false;
  return def->section->file == obj;
}

// Check whether this object participates in AMDGPU object-linking metadata.
template <class ELFT> static bool hasAMDGPUInfoSection(ObjFile<ELFT> *obj) {
  return obj->amdgpuInfoSectionIndex != 0;
}

// Parse one object's .amdgpu.info and .amdgpu.strtab sections into the shared
// call graph, direct resource records, and direct LDS/barrier uses.
template <class ELFT>
static void parseInfoSection(ObjFile<ELFT> *obj, AMDGPUCallGraph &cg,
                             const DenseMap<Symbol *, size_t> &ldsSymToIndex,
                             const DenseMap<Symbol *, size_t> &barSymToIndex) {
  if (!hasAMDGPUInfoSection(obj))
    return;

  ArrayRef<typename ELFT::Shdr> objSections = obj->template getELFShdrs<ELFT>();
  const auto &sec = objSections[obj->amdgpuInfoSectionIndex];
  ArrayRef<uint8_t> data = CHECK(obj->getObj().getSectionContents(sec),
                                 "could not read .amdgpu.info section");
  if (data.empty())
    return;

  DenseMap<uint64_t, RelocInfo> relocMap =
      buildRelocMap(obj, obj->amdgpuInfoSectionIndex);
  DenseMap<std::pair<SectionBase *, uint64_t>, Symbol *> addrToSym =
      buildSymbolAddressMap(obj);

  StringRef strtab;
  if (obj->amdgpuStrtabSectionIndex != 0) {
    const auto &strtabSec = objSections[obj->amdgpuStrtabSectionIndex];
    ArrayRef<uint8_t> strtabData =
        CHECK(obj->getObj().getSectionContents(strtabSec),
              "could not read .amdgpu.strtab section");
    strtab = StringRef(reinterpret_cast<const char *>(strtabData.data()),
                       strtabData.size());
  }

  CGNode *curNode = nullptr;
  bool curFuncIsPrevailing = false;
  size_t pos = 0;
  while (pos + 2 <= data.size()) {
    uint8_t kind = data[pos];
    uint8_t len = data[pos + 1];
    size_t payloadOff = pos + 2;
    if (payloadOff + len > data.size())
      break;

    switch (kind) {
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_FUNC): {
      if (len < 8)
        break;
      Symbol *sym =
          resolveRelocSym(obj, relocMap, addrToSym, payloadOff, "func");
      if (!sym || sym->getName().empty()) {
        curNode = nullptr;
        curFuncIsPrevailing = false;
        break;
      }
      curNode = &cg.getOrCreate(sym);
      curFuncIsPrevailing = isPrevailingInfoFunction(obj, sym);
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_FLAGS): {
      if (!curNode || !curFuncIsPrevailing || len < 4)
        break;
      AMDGPU::FuncInfoFlags flags = static_cast<AMDGPU::FuncInfoFlags>(
          read32le(data.data() + payloadOff));
      curNode->localRes.usesVCC =
          !!(flags & AMDGPU::FuncInfoFlags::FUNC_USES_VCC);
      curNode->localRes.usesFlatScratch =
          !!(flags & AMDGPU::FuncInfoFlags::FUNC_USES_FLAT_SCRATCH);
      curNode->localRes.hasDynSizedStack =
          !!(flags & AMDGPU::FuncInfoFlags::FUNC_HAS_DYN_STACK);
      curNode->hasLocalRes = true;
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_NUM_VGPR): {
      if (!curNode || !curFuncIsPrevailing || len < 4)
        break;
      curNode->localRes.numArchVGPR = read32le(data.data() + payloadOff);
      curNode->hasLocalRes = true;
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_NUM_AGPR): {
      if (!curNode || !curFuncIsPrevailing || len < 4)
        break;
      curNode->localRes.numAccVGPR = read32le(data.data() + payloadOff);
      curNode->hasLocalRes = true;
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_NUM_SGPR): {
      if (!curNode || !curFuncIsPrevailing || len < 4)
        break;
      curNode->localRes.numSGPR = read32le(data.data() + payloadOff);
      curNode->hasLocalRes = true;
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_PRIVATE_SEGMENT_SIZE): {
      if (!curNode || !curFuncIsPrevailing || len < 4)
        break;
      curNode->localRes.scratchSize = read32le(data.data() + payloadOff);
      curNode->hasLocalRes = true;
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_OCCUPANCY): {
      if (!curNode || !curFuncIsPrevailing || len < 4)
        break;
      curNode->localRes.occupancy = read32le(data.data() + payloadOff);
      curNode->hasLocalRes = true;
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_WAVE_SIZE): {
      if (!curNode || !curFuncIsPrevailing || len < 4)
        break;
      curNode->localRes.waveSize = read32le(data.data() + payloadOff);
      curNode->hasLocalRes = true;
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_USE): {
      if (!curNode || !curFuncIsPrevailing || len < 8)
        break;
      Symbol *resSym =
          resolveRelocSym(obj, relocMap, addrToSym, payloadOff, "use");
      if (!resSym || resSym->getName().empty())
        break;
      auto barIt = barSymToIndex.find(resSym);
      if (barIt != barSymToIndex.end()) {
        LLVM_DEBUG(dbgs() << "  barrier use: " << curNode->sym->getName()
                          << " -> " << resSym->getName() << "\n");
        curNode->barrierUseIndices.push_back(barIt->second);
        cg.setHasBarrierUses();
        break;
      }
      auto ldsIt = ldsSymToIndex.find(resSym);
      if (ldsIt != ldsSymToIndex.end()) {
        LLVM_DEBUG(dbgs() << "  use: " << curNode->sym->getName() << " -> "
                          << resSym->getName() << "\n");
        curNode->ldsUseIndices.push_back(ldsIt->second);
        cg.setHasLDSUses();
      }
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_CALL): {
      if (!curNode || !curFuncIsPrevailing || len < 8)
        break;
      Symbol *dstSym =
          resolveRelocSym(obj, relocMap, addrToSym, payloadOff, "call");
      if (!dstSym || dstSym->getName().empty())
        break;
      CGNode &dst = cg.getOrCreate(dstSym);
      LLVM_DEBUG(dbgs() << "  call: " << curNode->sym->getName() << " -> "
                        << dstSym->getName() << "\n");
      curNode->callees.push_back(&dst);
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_INDIRECT_CALL): {
      if (!curNode || !curFuncIsPrevailing || len < 4)
        break;
      uint32_t typeIdOff = read32le(data.data() + payloadOff);
      StringRef typeId = getInfoString(strtab, typeIdOff);
      if (!typeId.empty()) {
        LLVM_DEBUG(dbgs() << "  indirect-call: " << curNode->sym->getName()
                          << " enc=" << typeId << "\n");
        cg.addIndirectCall(curNode, typeId);
      }
      break;
    }
    case static_cast<uint8_t>(AMDGPU::InfoKind::INFO_TYPEID): {
      if (!curNode || len < 4)
        break;
      cg.markAddressTaken(curNode);
      uint32_t typeIdOff = read32le(data.data() + payloadOff);
      StringRef typeId = getInfoString(strtab, typeIdOff);
      if (!typeId.empty()) {
        LLVM_DEBUG(dbgs() << "  signature: " << curNode->sym->getName()
                          << " typeId=" << typeId << "\n");
        cg.addSignature(curNode, typeId);
      }
      break;
    }
    default:
      LLVM_DEBUG(dbgs() << "  unknown info kind " << (unsigned)kind
                        << " (len=" << (unsigned)len << "), skipping\n");
      break;
    }

    pos = payloadOff + len;
  }

  // Emit debug summary of parsed resources.
  if (curNode && curNode->hasLocalRes) {
    LLVM_DEBUG({
      for (auto &[sym, node] : cg) {
        if (!node->hasLocalRes)
          continue;
        dbgs() << "  resource: " << sym->getName()
               << " vgpr=" << node->localRes.numArchVGPR
               << " agpr=" << node->localRes.numAccVGPR
               << " sgpr=" << node->localRes.numSGPR
               << " scratch=" << node->localRes.scratchSize
               << " occupancy=" << node->localRes.occupancy << "\n";
      }
    });
  }
}

// Kernels are identified by the companion kernel descriptor symbol emitted as
// <kernel>.kd.
static bool hasKernelDescriptor(Ctx &ctx, Symbol *sym) {
  std::string kdName = (sym->getName() + ".kd").str();
  Symbol *kdSym = ctx.symtab->find(kdName);
  Defined *kdDef = dyn_cast_or_null<Defined>(kdSym);
  return kdDef && kdDef->section;
}

// Mark graph nodes with kernel descriptors as roots for reachability and
// resource propagation.
static void markKernelsWithDescriptors(Ctx &ctx, AMDGPUCallGraph &cg) {
  for (auto &[sym, node] : cg) {
    if (hasKernelDescriptor(ctx, sym)) {
      LLVM_DEBUG(dbgs() << "  kernel: " << sym->getName() << "\n");
      cg.addKernel(*node);
    }
  }
}

//===----------------------------------------------------------------------===//
// Call graph SCC condensation, reachability, and resource propagation
//===----------------------------------------------------------------------===//

// Validate local resource metadata for a kernel-reachable node before resource
// propagation depends on it.
static bool validateResourceNode(Ctx &ctx, CGNode *node) {
  assert(node->hasLocalRes && "missing resource usage after alias resolution");

  if (node->localRes.occupancy == 0) {
    Err(ctx) << "AMDGPU: function '" << node->sym->getName()
             << "' has invalid ABI occupancy 0";
    LLVM_DEBUG(dbgs() << "    resolve " << node->sym->getName()
                      << " (invalid occupancy)\n");
    return false;
  }

  return true;
}

// Merge resource usage from another ResourceInfo into the result using
// element-wise max for register/scratch counts and OR for boolean flags.
static void mergeResourceInfo(ResourceInfo &result, const ResourceInfo &info) {
  result.numArchVGPR = std::max(result.numArchVGPR, info.numArchVGPR);
  result.numAccVGPR = std::max(result.numAccVGPR, info.numAccVGPR);
  result.numSGPR = std::max(result.numSGPR, info.numSGPR);
  result.scratchSize = std::max(result.scratchSize, info.scratchSize);
  result.usesVCC |= info.usesVCC;
  result.usesFlatScratch |= info.usesFlatScratch;
  result.hasDynSizedStack |= info.hasDynSizedStack;
}

bool AMDGPUCallGraph::buildCondensedGraph(Ctx &ctx, bool validateResources) {
  DenseMap<CGNode *, unsigned> nodeIndex;
  DenseMap<CGNode *, unsigned> lowLink;
  DenseSet<CGNode *> onStack;
  SmallVector<CGNode *, 16> stack;
  unsigned nextIndex = 0;

  // Start from kernels so resource diagnostics are limited to code that can
  // execute. A validation failure aborts the build; callers do not inspect the
  // partially constructed SCC state.
  for (CGNode *kernel : kernels()) {
    if (nodeIndex.count(kernel))
      continue;
    if (!buildSCCs(ctx, kernel, nodeIndex, lowLink, onStack, stack, nextIndex,
                   validateResources))
      return false;
  }

  buildSCCEdges();
  return true;
}

bool AMDGPUCallGraph::buildSCCs(Ctx &ctx, CGNode *node,
                                DenseMap<CGNode *, unsigned> &nodeIndex,
                                DenseMap<CGNode *, unsigned> &lowLink,
                                DenseSet<CGNode *> &onStack,
                                SmallVectorImpl<CGNode *> &stack,
                                unsigned &nextIndex, bool validateResources) {
  if (validateResources && !validateResourceNode(ctx, node))
    return false;

  nodeIndex[node] = nextIndex;
  lowLink[node] = nextIndex;
  ++nextIndex;
  stack.push_back(node);
  onStack.insert(node);

  if (validateResources) {
    const ResourceInfo &info = node->localRes;
    LLVM_DEBUG(dbgs() << "    resolve " << node->sym->getName()
                      << " (has local res: vgpr=" << info.numArchVGPR
                      << " sgpr=" << info.numSGPR
                      << " scratch=" << info.scratchSize << ")\n");
  }

  if (!node->callees.empty()) {
    LLVM_DEBUG({
      if (validateResources) {
        dbgs() << "    " << node->sym->getName() << " has "
               << node->callees.size() << " callees:";
        for (CGNode *c : node->callees)
          dbgs() << " " << c->sym->getName();
        dbgs() << "\n";
      }
    });

    for (CGNode *callee : node->callees) {
      if (validateResources) {
        if (!validateResourceNode(ctx, callee))
          return false;

        if (callee->localRes.occupancy < node->localRes.occupancy) {
          Err(ctx) << "AMDGPU: incompatible ABI occupancy: function '"
                   << node->sym->getName() << "' requires occupancy "
                   << node->localRes.occupancy << " but calls '"
                   << callee->sym->getName() << "' with occupancy "
                   << callee->localRes.occupancy;
          return false;
        }
      }

      auto calleeIndex = nodeIndex.find(callee);
      if (calleeIndex == nodeIndex.end()) {
        if (!buildSCCs(ctx, callee, nodeIndex, lowLink, onStack, stack,
                       nextIndex, validateResources))
          return false;
        lowLink[node] = std::min(lowLink[node], lowLink[callee]);
      } else if (onStack.count(callee)) {
        lowLink[node] = std::min(lowLink[node], calleeIndex->second);
      }
    }
  }

  if (lowLink[node] != nodeIndex[node])
    return true;

  unsigned sccId = sccs.size();
  sccs.emplace_back();
  CallGraphSCC &scc = sccs.back();

  for (;;) {
    CGNode *member = stack.pop_back_val();
    onStack.erase(member);
    scc.members.push_back(member);
    nodeToSCC[member] = sccId;

    if (validateResources)
      mergeResourceInfo(scc.localRes, member->localRes);

    for (size_t idx : member->ldsUseIndices)
      scc.reachableLDS.insert(idx);

    for (size_t idx : member->barrierUseIndices)
      scc.reachableBarriers.insert(idx);

    if (member == node)
      break;
  }

  return true;
}

void AMDGPUCallGraph::buildSCCEdges() {
  for (unsigned sccId = 0, e = sccs.size(); sccId != e; ++sccId) {
    CallGraphSCC &scc = sccs[sccId];
    if (scc.members.size() > 1) {
      scc.isRecursive = true;
      scc.localRes.hasDynSizedStack = true;
    }

    DenseSet<unsigned> seenCallees;
    for (CGNode *member : scc.members) {
      for (CGNode *callee : member->callees) {
        auto it = nodeToSCC.find(callee);
        assert(it != nodeToSCC.end() && "callee should be in call graph SCC");

        unsigned calleeSCC = it->second;
        if (calleeSCC == sccId) {
          scc.isRecursive = true;
          scc.localRes.hasDynSizedStack = true;
          continue;
        }

        assert(calleeSCC < sccId &&
               "sccs should be in reverse topological order");

        if (seenCallees.insert(calleeSCC).second)
          scc.callees.push_back(calleeSCC);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// LDS resolution
//===----------------------------------------------------------------------===//

void AMDGPUCallGraph::computeKernelReachability() {
  for (unsigned sccId = 0, e = sccs.size(); sccId != e; ++sccId) {
    CallGraphSCC &scc = sccs[sccId];
    for (unsigned calleeSCC : scc.callees) {
      assert(calleeSCC < sccId &&
             "sccs should be in reverse topological order");
      const CallGraphSCC &callee = sccs[calleeSCC];
      scc.reachableLDS.insert(callee.reachableLDS.begin(),
                              callee.reachableLDS.end());
      scc.reachableBarriers.insert(callee.reachableBarriers.begin(),
                                   callee.reachableBarriers.end());
    }
  }

  for (CGNode *kernel : kernels()) {
    auto it = nodeToSCC.find(kernel);
    assert(it != nodeToSCC.end() && "kernel should be in call graph SCC");
    const CallGraphSCC &scc = sccs[it->second];
    kernel->reachableLDS.clear();
    kernel->reachableLDS.insert(scc.reachableLDS.begin(),
                                scc.reachableLDS.end());
    kernel->reachableBarriers.clear();
    kernel->reachableBarriers.insert(scc.reachableBarriers.begin(),
                                     scc.reachableBarriers.end());
  }
}

// Order LDS indices using the same layout priority as the symbol array.
static bool compareLDSIndex(ArrayRef<LDSSymbolInfo> ldsSymbols, size_t a,
                            size_t b) {
  return compareLDSSymbol(ldsSymbols[a], ldsSymbols[b]);
}

// Find the current frontier shared by all kernels that use one LDS object.
static uint64_t
getMaxKernelOffset(ArrayRef<CGNode *> users,
                   const DenseMap<CGNode *, uint64_t> &kernelOffsets) {
  uint64_t offset = 0;
  for (CGNode *user : users) {
    auto it = kernelOffsets.find(user);
    assert(it != kernelOffsets.end() && "missing kernel offset");
    offset = std::max(offset, it->second);
  }
  return offset;
}

#ifndef NDEBUG
// Debug-only consistency checks for the grouped LDS layout.
static void assertLDSLayoutInvariants(
    ArrayRef<LDSSymbolInfo> ldsSymbols, ArrayRef<CGNode *> kernels,
    const DenseMap<size_t, SmallVector<CGNode *, 4>> &ldsToUsers,
    const DenseSet<size_t> &assigned) {
  assert(assigned.size() == ldsSymbols.size() &&
         "not all LDS symbols assigned");
  for (size_t idx = 0, e = ldsSymbols.size(); idx < e; ++idx)
    assert(assigned.count(idx) && "LDS symbol not assigned");

  for (const auto &[idx, users] : ldsToUsers) {
    uint64_t offset = ldsSymbols[idx].assignedOffset;
    for (CGNode *user : users) {
      assert(user->reachableLDS.count(idx) && "inconsistent LDS user map");
      assert(ldsSymbols[idx].assignedOffset == offset &&
             "LDS has non-uniform offset across kernels");
    }
  }

  for (CGNode *kernel : kernels) {
    SmallVector<std::pair<uint64_t, uint64_t>, 8> intervals;
    for (size_t idx : kernel->reachableLDS) {
      if (ldsSymbols[idx].size == 0)
        continue;
      uint64_t begin = ldsSymbols[idx].assignedOffset;
      intervals.push_back({begin, begin + ldsSymbols[idx].size});
    }

    llvm::sort(intervals);
    for (size_t i = 1, e = intervals.size(); i < e; ++i)
      assert(intervals[i - 1].second <= intervals[i].first &&
             "kernel has overlapping LDS intervals");
  }
}
#endif

// Assign offsets using a per-kernel frontier. Each LDS still receives one
// global offset, but only kernels that use that LDS advance their current
// frontier. This lets LDS objects used by disjoint kernel sets overlap even
// when other LDS objects connect those kernels transitively.
static void assignGroupedOffsets(SmallVectorImpl<LDSSymbolInfo> &ldsSymbols,
                                 AMDGPUCallGraph &cg) {
  if (!cg.hasLDSUses()) {
    assignUniversalOffsets(ldsSymbols);
    return;
  }

  SmallVector<CGNode *, 8> kernels(cg.kernels().begin(), cg.kernels().end());
  DenseMap<size_t, SmallVector<CGNode *, 4>> ldsToUsers;
  DenseMap<CGNode *, uint64_t> kernelOffsets;
  for (CGNode *kernel : kernels) {
    kernelOffsets[kernel] = 0;
    for (size_t idx : kernel->reachableLDS)
      ldsToUsers[idx].push_back(kernel);
  }

  auto idxCmp = [&](size_t a, size_t b) {
    return compareLDSIndex(ldsSymbols, a, b);
  };
  auto getUserCount = [&](size_t idx) {
    auto it = ldsToUsers.find(idx);
    return it == ldsToUsers.end() ? 0 : it->second.size();
  };
  auto claimedCmp = [&](size_t a, size_t b) {
    size_t aUsers = getUserCount(a);
    size_t bUsers = getUserCount(b);
    if (aUsers != bUsers)
      return aUsers > bUsers;
    return idxCmp(a, b);
  };

  DenseSet<size_t> assigned;
  SmallVector<size_t, 8> fixedLDS;
  SmallVector<size_t, 4> dynamicLDS;
  SmallVector<size_t, 8> unclaimed;
  SmallVector<size_t, 4> unclaimedDyn;

  for (size_t i = 0, e = ldsSymbols.size(); i < e; ++i) {
    bool hasUsers = getUserCount(i) != 0;
    if (hasUsers && ldsSymbols[i].size != 0)
      fixedLDS.push_back(i);
    else if (hasUsers)
      dynamicLDS.push_back(i);
    else if (ldsSymbols[i].size == 0)
      unclaimedDyn.push_back(i);
    else
      unclaimed.push_back(i);
  }

  llvm::sort(fixedLDS, claimedCmp);
  llvm::sort(dynamicLDS, claimedCmp);

  for (size_t idx : fixedLDS) {
    ArrayRef<CGNode *> users = ldsToUsers[idx];
    uint64_t offset = alignTo(getMaxKernelOffset(users, kernelOffsets),
                              ldsSymbols[idx].alignment);
    ldsSymbols[idx].assignedOffset = offset;
    [[maybe_unused]] bool inserted = assigned.insert(idx).second;
    assert(inserted && "LDS symbol assigned multiple times");

    uint64_t nextOffset = offset + ldsSymbols[idx].size;
    LLVM_DEBUG(dbgs() << "    allocate " << ldsSymbols[idx].sym->getName()
                      << " at " << offset << " size=" << ldsSymbols[idx].size
                      << " users=" << users.size() << "\n");
    for (CGNode *user : users)
      kernelOffsets[user] = nextOffset;
  }

  for (size_t idx : dynamicLDS) {
    ArrayRef<CGNode *> users = ldsToUsers[idx];
    uint64_t offset = alignTo(getMaxKernelOffset(users, kernelOffsets),
                              ldsSymbols[idx].alignment);
    ldsSymbols[idx].assignedOffset = offset;
    [[maybe_unused]] bool inserted = assigned.insert(idx).second;
    assert(inserted && "LDS symbol assigned multiple times");
  }

  // LDS symbols not reachable from any kernel (e.g. dead device functions,
  // unused library symbols, or links with no kernels) still need valid
  // addresses for relocation resolution. Assign them offsets from 0.
  if (!unclaimed.empty() || !unclaimedDyn.empty()) {
    llvm::sort(unclaimed, idxCmp);
    llvm::sort(unclaimedDyn, idxCmp);
    uint64_t offset = 0;
    for (size_t idx : unclaimed) {
      offset = alignTo(offset, ldsSymbols[idx].alignment);
      ldsSymbols[idx].assignedOffset = offset;
      [[maybe_unused]] bool inserted = assigned.insert(idx).second;
      assert(inserted && "LDS symbol assigned multiple times");
      offset += ldsSymbols[idx].size;
    }
    if (!unclaimedDyn.empty()) {
      Align maxDynAlign(1);
      for (size_t idx : unclaimedDyn)
        maxDynAlign = std::max(maxDynAlign, Align(ldsSymbols[idx].alignment));
      uint64_t dynBase = alignTo(offset, maxDynAlign);
      for (size_t idx : unclaimedDyn) {
        ldsSymbols[idx].assignedOffset = dynBase;
        [[maybe_unused]] bool inserted = assigned.insert(idx).second;
        assert(inserted && "LDS symbol assigned multiple times");
      }
    }
  }

#ifndef NDEBUG
  assertLDSLayoutInvariants(ldsSymbols, kernels, ldsToUsers, assigned);
#endif
}

// Compute each kernel's fixed LDS usage from the final offsets of the LDS
// objects reachable from that kernel.
static void computeKernelLDSSizes(ArrayRef<LDSSymbolInfo> ldsSymbols,
                                  AMDGPUCallGraph &cg) {
  if (!cg.hasLDSUses()) {
    uint32_t totalSize = 0;
    for (const LDSSymbolInfo &lds : ldsSymbols)
      totalSize = std::max<uint64_t>(totalSize, lds.assignedOffset + lds.size);
    for (CGNode *kernel : cg.kernels())
      kernel->ldsSize = totalSize;
    return;
  }

  for (CGNode *kernel : cg.kernels()) {
    uint32_t maxEnd = 0;
    for (size_t idx : kernel->reachableLDS)
      maxEnd = std::max<uint64_t>(maxEnd, ldsSymbols[idx].assignedOffset +
                                              ldsSymbols[idx].size);
    kernel->ldsSize = maxEnd;
  }
}

// Assign LDS offsets, rewrite LDS symbols to those offsets, and record each
// kernel's group segment size.
static void resolveLDS(Ctx &ctx, SmallVectorImpl<LDSSymbolInfo> &ldsSymbols,
                       AMDGPUCallGraph &cg) {
  LLVM_DEBUG(dbgs() << "AMDGPU LDS: assigning grouped offsets\n");
  assignGroupedOffsets(ldsSymbols, cg);
  computeKernelLDSSizes(ldsSymbols, cg);

  // Symbol::overwrite preserves the old symbol's visibility, so for shared-
  // object links (AMDGPU code objects use -shared) we must explicitly force
  // STV_HIDDEN to prevent the symbol from being preemptible, which would cause
  // R_AMDGPU_ABS32_LO relocations to be rejected by the relocation scanner.
  LLVM_DEBUG(dbgs() << "AMDGPU LDS: final symbol assignments:\n");
  for (const LDSSymbolInfo &lds : ldsSymbols) {
    LLVM_DEBUG(dbgs() << "  " << lds.sym->getName() << " -> offset="
                      << lds.assignedOffset << " size=" << lds.size << "\n");
    Defined(ctx, ctx.internalFile, lds.sym->getName(), STB_GLOBAL, STV_HIDDEN,
            STT_NOTYPE, lds.assignedOffset, lds.size, nullptr)
        .overwrite(*lds.sym);
    if (ctx.arg.shared)
      lds.sym->stOther = (lds.sym->stOther & ~3) | STV_HIDDEN;
  }

  LLVM_DEBUG({
    dbgs() << "AMDGPU LDS: per-kernel LDS sizes:\n";
    for (CGNode *kernel : cg.kernels())
      dbgs() << "  " << kernel->sym->getName() << " -> " << kernel->ldsSize
             << " bytes\n";
  });
}

//===----------------------------------------------------------------------===//
// Named barrier resolution
//===----------------------------------------------------------------------===//

// Assign named-barrier hardware IDs within groups of kernels that can reach the
// same barrier, then rewrite named-barrier symbols to encoded barrier
// addresses.
static void resolveNamedBarriers(
    Ctx &ctx, SmallVectorImpl<NamedBarrierInfo> &barriers, AMDGPUCallGraph &cg,
    const DenseMap<size_t, SmallVector<CGNode *, 2>> &barToUsers) {
  auto isKernelTier = [&](size_t barIdx) {
    auto it = barToUsers.find(barIdx);
    if (it == barToUsers.end())
      return false;
    return it->second.size() == 1 && it->second[0]->isKernel;
  };

  EquivalenceClasses<CGNode *> groups;
  for (CGNode *k : cg.kernels())
    groups.insert(k);

  for (size_t barIdx = 0, e = barriers.size(); barIdx < e; ++barIdx) {
    SmallVector<CGNode *, 4> reachingKernels;
    for (CGNode *k : cg.kernels()) {
      if (k->reachableBarriers.count(barIdx))
        reachingKernels.push_back(k);
    }
    for (size_t i = 1; i < reachingKernels.size(); ++i)
      groups.unionSets(reachingKernels[0], reachingKernels[i]);
  }

  [[maybe_unused]] unsigned groupIdx = 0;
  for (auto it = groups.begin(), e = groups.end(); it != e; ++it) {
    if (!(*it)->isLeader())
      continue;

    DenseSet<size_t> groupBarriers;
    SmallVector<CGNode *, 4> groupKernels;
    LLVM_DEBUG(dbgs() << "  barrier group " << groupIdx << " kernels:");
    for (auto mi = groups.member_begin(**it); mi != groups.member_end(); ++mi) {
      LLVM_DEBUG(dbgs() << " " << (*mi)->sym->getName());
      groupKernels.push_back(*mi);
      groupBarriers.insert((*mi)->reachableBarriers.begin(),
                           (*mi)->reachableBarriers.end());
    }
    LLVM_DEBUG(dbgs() << "\n");
    ++groupIdx;

    SmallVector<size_t> sharedTier, kernelTier;
    for (size_t idx : groupBarriers) {
      if (isKernelTier(idx))
        kernelTier.push_back(idx);
      else
        sharedTier.push_back(idx);
    }

    llvm::sort(sharedTier, [&](size_t a, size_t b) {
      return barriers[a].sym->getName() < barriers[b].sym->getName();
    });
    llvm::sort(kernelTier, [&](size_t a, size_t b) {
      return barriers[a].sym->getName() < barriers[b].sym->getName();
    });

    uint32_t nextBarId = 1;
    for (size_t idx : sharedTier) {
      barriers[idx].assignedBarId = nextBarId;
      nextBarId += barriers[idx].slotCount;
    }
    for (size_t idx : kernelTier) {
      barriers[idx].assignedBarId = nextBarId;
      nextBarId += barriers[idx].slotCount;
    }

    if (nextBarId - 1 > 31) {
      SmallString<256> kernelList;
      for (size_t i = 0; i < groupKernels.size(); ++i) {
        if (i > 0)
          kernelList += ", ";
        kernelList += groupKernels[i]->sym->getName();
      }
      Err(ctx) << "AMDGPU: named barrier ID overflow (max ID "
               << (nextBarId - 1)
               << " exceeds limit of 31) in kernel group: " << kernelList;
    }
  }

  constexpr uint32_t barScope = 0; // BARRIER_SCOPE_WORKGROUP
  LLVM_DEBUG(dbgs() << "AMDGPU Named Barriers: final assignments:\n");
  for (NamedBarrierInfo &bar : barriers) {
    if (bar.assignedBarId == 0)
      continue;
    uint32_t addr = 0x802000u | (barScope << 9) | (bar.assignedBarId << 4);
    LLVM_DEBUG(dbgs() << "  " << bar.sym->getName()
                      << " -> barId=" << bar.assignedBarId << " addr=0x"
                      << Twine::utohexstr(addr) << "\n");
    Defined(ctx, ctx.internalFile, bar.sym->getName(), STB_GLOBAL, STV_HIDDEN,
            STT_NOTYPE, addr, 0, nullptr)
        .overwrite(*bar.sym);
    if (ctx.arg.shared)
      bar.sym->stOther = (bar.sym->stOther & ~3) | STV_HIDDEN;
  }

  for (CGNode *kernel : cg.kernels()) {
    uint32_t maxBarEnd = 0;
    for (size_t idx : kernel->reachableBarriers) {
      uint32_t end = barriers[idx].assignedBarId + barriers[idx].slotCount - 1;
      maxBarEnd = std::max(maxBarEnd, end);
    }
    kernel->numNamedBarrier = maxBarEnd;
    LLVM_DEBUG(dbgs() << "  " << kernel->sym->getName()
                      << " numNamedBarrier=" << maxBarEnd << "\n");
  }
}

// Check that each kernel's final LDS usage satisfies the ABI occupancy recorded
// in .amdgpu.info.
static bool validateKernelLDSOccupancy(Ctx &ctx, AMDGPUCallGraph &cg,
                                       const MCSubtargetInfo &sti) {
  for (CGNode *kernel : cg.kernels()) {
    assert(kernel->hasLocalRes &&
           "kernel resource usage should have been validated");
    assert(kernel->localRes.occupancy != 0 &&
           "kernel occupancy should have been validated");
    if (AMDGPU::IsaInfo::isLocalMemorySizeCompatibleWithOccupancy(
            sti, kernel->ldsSize, kernel->localRes.occupancy))
      continue;

    Err(ctx) << "AMDGPU: kernel '" << kernel->sym->getName() << "' uses "
             << kernel->ldsSize << " bytes of LDS, which does not meet ABI "
             << "occupancy " << kernel->localRes.occupancy;
    return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Resource usage propagation
//===----------------------------------------------------------------------===//

void AMDGPUCallGraph::propagateResourceUsage() {
  for (unsigned sccId = 0, e = sccs.size(); sccId != e; ++sccId) {
    CallGraphSCC &scc = sccs[sccId];
    ResourceInfo result = scc.localRes;
    uint32_t maxCalleeScratch = 0;
    for (unsigned calleeSCC : scc.callees) {
      assert(calleeSCC < sccId &&
             "sccs should be in reverse topological order");
      const CallGraphSCC &callee = sccs[calleeSCC];
      assert(callee.hasPropagatedRes && "callee resource should be propagated");
      mergeResourceInfo(result, callee.propagatedRes);
      maxCalleeScratch =
          std::max(maxCalleeScratch, callee.propagatedRes.scratchSize);
    }
    // For recursive SCCs, scratchSize remains the finite fixed-frame
    // contribution. The unbounded recursive depth is represented by
    // hasDynSizedStack, which forces dynamic stack handling.
    result.scratchSize = scc.localRes.scratchSize + maxCalleeScratch;

    scc.propagatedRes = result;
    scc.hasPropagatedRes = true;
    for (CGNode *member : scc.members) {
      member->propagatedRes = result;
      member->hasPropagatedRes = true;
    }
  }

  for (CGNode *kernel : kernels()) {
    LLVM_DEBUG(dbgs() << "  propagated " << kernel->sym->getName()
                      << ": vgpr=" << kernel->propagatedRes.numArchVGPR
                      << " agpr=" << kernel->propagatedRes.numAccVGPR
                      << " sgpr=" << kernel->propagatedRes.numSGPR
                      << " scratch=" << kernel->propagatedRes.scratchSize
                      << " dynstack=" << kernel->propagatedRes.hasDynSizedStack
                      << "\n");
  }
}

namespace {
// Kernel resource values after applying all linker-side propagation and
// target-specific derived counts. Both kernel descriptors and HSA metadata
// consume this so their formulas cannot drift.
struct ResolvedKernelResources {
  uint32_t totalVGPR = 0;
  uint32_t totalSGPR = 0;
  uint32_t numAccVGPR = 0;
  uint32_t scratchSize = 0;
  uint32_t accumOffset = 0;
  uint32_t namedBarCnt = 0;
  bool scratchEnable = false;
  bool usesDynamicStack = false;
};
} // namespace

// Resolve all resource fields that are independent of the existing descriptor
// contents. VGPR block encoding is left to the KD patcher because it depends on
// the per-descriptor ENABLE_WAVEFRONT_SIZE32 bit.
static ResolvedKernelResources
resolveKernelResources(const CGNode &kernel, const MCSubtargetInfo &sti) {
  assert(kernel.hasPropagatedRes &&
         "kernel resource usage should have been propagated");

  const ResourceInfo &info = kernel.propagatedRes;
  ResolvedKernelResources res;
  res.totalVGPR = AMDGPU::getTotalNumVGPRs(AMDGPU::isGFX90A(sti),
                                           info.numAccVGPR, info.numArchVGPR);
  res.totalSGPR = info.numSGPR + AMDGPU::IsaInfo::getNumExtraSGPRs(
                                     sti, info.usesVCC, info.usesFlatScratch);
  res.numAccVGPR = info.numAccVGPR;
  res.scratchSize = info.scratchSize;
  unsigned archGranule = AMDGPU::IsaInfo::getArchVGPRAllocGranule();
  res.accumOffset = divideCeil(std::max(info.numArchVGPR, 1u), archGranule) - 1;
  res.namedBarCnt = divideCeil(kernel.numNamedBarrier, 4);
  res.scratchEnable = res.scratchSize > 0 || info.hasDynSizedStack;
  res.usesDynamicStack = info.hasDynSizedStack;
  return res;
}

// Patch kernel descriptors in-place with resolved LDS, named-barrier, and
// propagated resource fields.
static void patchKernelDescriptors(Ctx &ctx, AMDGPUCallGraph &cg,
                                   const MCSubtargetInfo *sti, bool hasLDS,
                                   bool hasBarriers) {
  using namespace llvm::amdhsa;
  bool hasAccumOffset = AMDGPU::isGFX90A(*sti);
  bool sgprBlocksAlwaysZero = AMDGPU::isGFX10Plus(*sti);
  bool hasNamedBarCnt = AMDGPU::isGFX1250Plus(*sti);

  // Track sections that have been copied to writable memory so we don't
  // allocate redundant copies when multiple KDs share the same section.
  DenseSet<InputSection *> copiedSections;

  for (CGNode *kernel : cg.kernels()) {
    StringRef name = kernel->sym->getName();
    std::string kdName = (name + ".kd").str();
    Symbol *kdSym = ctx.symtab->find(kdName);
    if (!kdSym)
      continue;
    auto *kdDef = dyn_cast<Defined>(kdSym);
    if (!kdDef || !kdDef->section)
      continue;
    auto *isec = dyn_cast<InputSection>(kdDef->section);
    if (!isec)
      continue;

    uint64_t off = kdDef->value;
    if (off + sizeof(kernel_descriptor_t) > isec->size)
      continue;

    // The section content may be in read-only mmap'd memory. Make a writable
    // copy the first time we need to patch a KD in this section.
    if (copiedSections.insert(isec).second) {
      auto *newBuf = ctx.bAlloc.Allocate<uint8_t>(isec->size);
      memcpy(newBuf, isec->content_, isec->size);
      isec->content_ = newBuf;
    }

    auto *buf = const_cast<uint8_t *>(isec->content_) + off;

    if (hasLDS) {
      write32le(buf + GROUP_SEGMENT_FIXED_SIZE_OFFSET, kernel->ldsSize);
      LLVM_DEBUG(dbgs() << "  patched " << name
                        << ".kd group_segment_fixed_size = " << kernel->ldsSize
                        << "\n");
    }

    if (!kernel->hasPropagatedRes)
      continue;

    ResolvedKernelResources res = resolveKernelResources(*kernel, *sti);
    write32le(buf + PRIVATE_SEGMENT_FIXED_SIZE_OFFSET, res.scratchSize);

    // Read the per-kernel ENABLE_WAVEFRONT_SIZE32 bit from the KD -- it
    // affects the VGPR encoding granule on GFX10+.
    uint16_t kcp = read16le(buf + KERNEL_CODE_PROPERTIES_OFFSET);
    bool enableWave32 = kcp & KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;

    // Patch compute_pgm_rsrc1: preserve constant bits, replace VGPR/SGPR blocks
    uint32_t rsrc1 = read32le(buf + COMPUTE_PGM_RSRC1_OFFSET);
    uint32_t vgprBlocks = AMDGPU::IsaInfo::getEncodedNumVGPRBlocks(
        *sti, res.totalVGPR, enableWave32);
    uint32_t sgprBlocks =
        sgprBlocksAlwaysZero
            ? 0
            : AMDGPU::IsaInfo::getNumSGPRBlocks(*sti, res.totalSGPR);
    rsrc1 &= ~COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT;
    rsrc1 |=
        (vgprBlocks << COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT) &
        COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT;
    rsrc1 &= ~COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT;
    rsrc1 |= (sgprBlocks
              << COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT) &
             COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT;
    write32le(buf + COMPUTE_PGM_RSRC1_OFFSET, rsrc1);

    // Patch compute_pgm_rsrc2: update scratch enable bit
    uint32_t rsrc2 = read32le(buf + COMPUTE_PGM_RSRC2_OFFSET);
    rsrc2 &= ~COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT;
    if (res.scratchEnable)
      rsrc2 |= COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT;
    write32le(buf + COMPUTE_PGM_RSRC2_OFFSET, rsrc2);

    // Patch compute_pgm_rsrc3: update AccumOffset for GFX90A
    if (hasAccumOffset) {
      uint32_t rsrc3 = read32le(buf + COMPUTE_PGM_RSRC3_OFFSET);
      rsrc3 &= ~COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET;
      rsrc3 |=
          (res.accumOffset << COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET_SHIFT) &
          COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET;
      write32le(buf + COMPUTE_PGM_RSRC3_OFFSET, rsrc3);
    }

    // Patch compute_pgm_rsrc3: update NAMED_BAR_CNT for GFX1250
    if (hasBarriers && hasNamedBarCnt) {
      uint32_t rsrc3 = read32le(buf + COMPUTE_PGM_RSRC3_OFFSET);
      rsrc3 &= ~COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT;
      rsrc3 |=
          (res.namedBarCnt << COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT_SHIFT) &
          COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT;
      write32le(buf + COMPUTE_PGM_RSRC3_OFFSET, rsrc3);
      LLVM_DEBUG(dbgs() << "  patched " << name
                        << ".kd NAMED_BAR_CNT=" << res.namedBarCnt << "\n");
    }

    // Patch kernel_code_properties: update USES_DYNAMIC_STACK
    kcp &= ~KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK;
    if (res.usesDynamicStack)
      kcp |= KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK;
    write16le(buf + KERNEL_CODE_PROPERTIES_OFFSET, kcp);

    LLVM_DEBUG(dbgs() << "  patched " << name << ".kd: scratch="
                      << res.scratchSize << " vgprBlocks=" << vgprBlocks
                      << " sgprBlocks=" << sgprBlocks
                      << " dynstack=" << res.usesDynamicStack << "\n");
  }
}

//===----------------------------------------------------------------------===//
// HSA metadata patching
//===----------------------------------------------------------------------===//

// Rewrite AMDHSA metadata notes so the YAML/msgpack-visible kernel resource
// fields match the patched kernel descriptors.
template <class ELFT>
static void patchHSAMetadata(Ctx &ctx, AMDGPUCallGraph &cg,
                             const MCSubtargetInfo *sti, bool hasLDS) {
  bool hasRes = false;
  for (CGNode *k : cg.kernels())
    if (k->hasPropagatedRes) {
      hasRes = true;
      break;
    }
  if (!hasLDS && !hasRes)
    return;

  DenseMap<StringRef, CGNode *> nameToKernel;
  for (CGNode *kernel : cg.kernels())
    nameToKernel[kernel->sym->getName()] = kernel;

  for (InputSectionBase *sec : ctx.inputSections) {
    auto *isec = dyn_cast<InputSection>(sec);
    if (!isec || isec->type != SHT_NOTE)
      continue;
    if (isec->name != ".note")
      continue;

    ArrayRef<uint8_t> data = isec->contentMaybeDecompress();
    if (data.size() < 12)
      continue;

    uint32_t nameSize = read32le(data.data());
    uint32_t descSize = read32le(data.data() + 4);
    uint32_t noteType = read32le(data.data() + 8);

    // NT_AMDGPU_METADATA = 32
    if (noteType != 32)
      continue;

    uint32_t nameOff = 12;
    uint32_t namePadded = alignTo(nameSize, 4);
    uint32_t descOff = nameOff + namePadded;

    if (descOff + descSize > data.size())
      continue;

    ArrayRef<uint8_t> msgpackData = data.slice(descOff, descSize);
    msgpack::Document doc;
    if (!doc.readFromBlob(
            StringRef(reinterpret_cast<const char *>(msgpackData.data()),
                      msgpackData.size()),
            false))
      continue;

    msgpack::MapDocNode root = doc.getRoot().getMap();
    msgpack::DocNode kernelsNode = root["amdhsa.kernels"];
    if (kernelsNode.isEmpty())
      continue;

    bool modified = false;
    msgpack::ArrayDocNode kernelsArray = kernelsNode.getArray();
    for (size_t i = 0, e = kernelsArray.size(); i < e; ++i) {
      msgpack::MapDocNode kernMap = kernelsArray[i].getMap();
      msgpack::DocNode nameNode = kernMap[".name"];
      if (nameNode.isEmpty())
        continue;

      StringRef kernName = nameNode.getString();
      auto it = nameToKernel.find(kernName);
      if (it == nameToKernel.end())
        continue;
      CGNode *kernel = it->second;

      if (hasLDS) {
        kernMap[".group_segment_fixed_size"] = doc.getNode(kernel->ldsSize);
        modified = true;
      }

      if (kernel->hasPropagatedRes) {
        ResolvedKernelResources res = resolveKernelResources(*kernel, *sti);
        kernMap[".sgpr_count"] = doc.getNode(res.totalSGPR);
        kernMap[".vgpr_count"] = doc.getNode(res.totalVGPR);
        kernMap[".agpr_count"] = doc.getNode(res.numAccVGPR);
        kernMap[".private_segment_fixed_size"] = doc.getNode(res.scratchSize);
        kernMap[".uses_dynamic_stack"] = doc.getNode(res.usesDynamicStack);
        modified = true;
      }
    }

    if (!modified)
      continue;

    std::string newMsgpack;
    doc.writeToBlob(newMsgpack);

    uint32_t newDescSize = newMsgpack.size();
    uint32_t newDescPadded = alignTo(newDescSize, 4);
    uint32_t newSize = nameOff + namePadded + newDescPadded;

    auto *buf = ctx.bAlloc.Allocate<uint8_t>(newSize);
    memset(buf, 0, newSize);
    write32le(buf, nameSize);
    write32le(buf + 4, newDescSize);
    write32le(buf + 8, noteType);
    memcpy(buf + nameOff, data.data() + nameOff, namePadded);
    memcpy(buf + nameOff + namePadded, newMsgpack.data(), newMsgpack.size());

    isec->content_ = buf;
    isec->size = newSize;
  }
}

//===----------------------------------------------------------------------===//
// Pre-link compatibility checks
//===----------------------------------------------------------------------===//

// Verify that all input files participating in object linking have compatible
// target IDs (GPU architecture and target features). Files with different mach
// types or incompatible xnack/sramecc settings cannot be linked.
template <class ELFT> static bool validateTargetID(Ctx &ctx) {
  ObjFile<ELFT> *firstObj = nullptr;
  uint32_t firstMach = 0;
  uint32_t firstXnack = 0;
  uint32_t firstSramEcc = 0;

  for (ELFFileBase *file : ctx.objectFiles) {
    auto *obj = cast<ObjFile<ELFT>>(file);
    if (!hasAMDGPUInfoSection(obj))
      continue;

    uint32_t eflags = obj->getObj().getHeader().e_flags;
    uint32_t mach = eflags & ELF::EF_AMDGPU_MACH;
    uint32_t xnack = eflags & ELF::EF_AMDGPU_FEATURE_XNACK_V4;
    uint32_t sramEcc = eflags & ELF::EF_AMDGPU_FEATURE_SRAMECC_V4;

    if (!firstObj) {
      firstObj = obj;
      firstMach = mach;
      firstXnack = xnack;
      firstSramEcc = sramEcc;
      continue;
    }

    if (mach != firstMach) {
      Err(ctx) << "AMDGPU object linking: incompatible GPU architecture "
               << "between " << firstObj->getName() << " and "
               << obj->getName();
      return false;
    }

    if (xnack != firstXnack && xnack != ELF::EF_AMDGPU_FEATURE_XNACK_ANY_V4 &&
        firstXnack != ELF::EF_AMDGPU_FEATURE_XNACK_ANY_V4) {
      Err(ctx) << "AMDGPU object linking: incompatible xnack setting "
               << "between " << firstObj->getName() << " and "
               << obj->getName();
      return false;
    }

    if (sramEcc != firstSramEcc &&
        sramEcc != ELF::EF_AMDGPU_FEATURE_SRAMECC_ANY_V4 &&
        firstSramEcc != ELF::EF_AMDGPU_FEATURE_SRAMECC_ANY_V4) {
      Err(ctx) << "AMDGPU object linking: incompatible sramecc setting "
               << "between " << firstObj->getName() << " and "
               << obj->getName();
      return false;
    }
  }

  return true;
}

// Validate wave size compatibility across call edges. A caller and callee must
// use the same wavefront size because the calling convention, register layout,
// and instruction semantics differ between wave32 and wave64.
static bool validateWaveSizeCompatibility(Ctx &ctx, AMDGPUCallGraph &cg) {
  bool valid = true;
  for (auto &[sym, node] : cg) {
    // Alias nodes remain in the graph map after resolveAliases, but all edges
    // to them have been redirected to the canonical node with resource info.
    if (!node->hasLocalRes)
      continue;
    for (CGNode *callee : node->callees) {
      assert(callee->hasLocalRes && "missing local resource info for callee");
      if (node->localRes.waveSize != callee->localRes.waveSize) {
        Err(ctx) << "AMDGPU object linking: wave size mismatch in call from '"
                 << node->sym->getName() << "' (wave" << node->localRes.waveSize
                 << ") to '" << callee->sym->getName() << "' (wave"
                 << callee->localRes.waveSize << ")";
        valid = false;
      }
    }
  }
  return valid;
}

//===----------------------------------------------------------------------===//
// Main entry point
//===----------------------------------------------------------------------===//

// Resolve AMDGPU object-linking metadata for one ELF class. This is run after
// regular symbol resolution so all referenced symbols and kernel descriptors
// are available for graph construction and patching.
template <class ELFT> void elf::resolveAMDGPUObjectLinking(Ctx &ctx) {
  llvm::TimeTraceScope timeScope("Resolve AMDGPU Object Linking");

  LLVM_DEBUG(dbgs() << "AMDGPU: validating target ID compatibility\n");
  if (!validateTargetID<ELFT>(ctx))
    return;

  LLVM_DEBUG(dbgs() << "AMDGPU: collecting LDS symbols\n");
  SmallVector<LDSSymbolInfo, 16> ldsSymbols;
  SmallVector<NamedBarrierInfo, 4> barriers;
  collectLDSSymbols(ctx, ldsSymbols, barriers);
  LLVM_DEBUG(dbgs() << "AMDGPU: found " << ldsSymbols.size() << " LDS symbols, "
                    << barriers.size() << " named barriers\n");

  // Build sym->index maps once (used by section parsers).
  DenseMap<Symbol *, size_t> ldsSymToIndex;
  for (size_t i = 0, e = ldsSymbols.size(); i < e; ++i)
    ldsSymToIndex[ldsSymbols[i].sym] = i;
  DenseMap<Symbol *, size_t> barSymToIndex;
  for (size_t i = 0, e = barriers.size(); i < e; ++i)
    barSymToIndex[barriers[i].sym] = i;

  AMDGPUCallGraph cg;

  LLVM_DEBUG(dbgs() << "AMDGPU: parsing sections\n");
  bool hasResourceUsage = false;
  for (ELFFileBase *file : ctx.objectFiles) {
    auto *obj = cast<ObjFile<ELFT>>(file);
    parseInfoSection(obj, cg, ldsSymToIndex, barSymToIndex);
    if (hasAMDGPUInfoSection(obj))
      hasResourceUsage = true;
  }
  LLVM_DEBUG(dbgs() << "AMDGPU: building indirect call edges\n");
  cg.buildIndirectEdges();

  LLVM_DEBUG(dbgs() << "AMDGPU: resolving function aliases\n");
  cg.resolveAliases();

  LLVM_DEBUG(dbgs() << "AMDGPU: validating wave size compatibility\n");
  if (!validateWaveSizeCompatibility(ctx, cg))
    return;

  LLVM_DEBUG(dbgs() << "AMDGPU: identifying kernels\n");
  markKernelsWithDescriptors(ctx, cg);

  // Build the named-barrier reverse map (barrier index -> direct user nodes)
  // for resolveNamedBarriers.
  DenseMap<size_t, SmallVector<CGNode *, 2>> barToUsers;
  for (auto &[sym, node] : cg) {
    for (size_t idx : node->barrierUseIndices)
      barToUsers[idx].push_back(node);
  }

  // Detect resource usage from any node in the call graph.
  if (!hasResourceUsage)
    hasResourceUsage = cg.begin() != cg.end();

  LLVM_DEBUG(dbgs() << "AMDGPU: " << ldsSymbols.size() << " regular LDS, "
                    << barriers.size() << " named barriers\n");
  LLVM_DEBUG(dbgs() << "AMDGPU: " << cg.kernels().size() << " kernels\n");
  LLVM_DEBUG(if (hasResourceUsage) dbgs()
             << "AMDGPU: has resource usage data\n");

  if (ldsSymbols.empty() && barriers.empty() && !hasResourceUsage) {
    LLVM_DEBUG(dbgs() << "AMDGPU: nothing to resolve\n");
    return;
  }

  // Construct MCSubtargetInfo from merged ELF e_flags for target-aware
  // resource computation (VGPR totals, extra SGPRs, etc.).
  uint32_t eflags = ctx.arg.eflags;
  StringRef cpu = AMDGPU::getArchNameFromElfMach(eflags & ELF::EF_AMDGPU_MACH);
  std::string features;
  auto addFeature = [&](StringRef feature) {
    if (!features.empty())
      features += ",";
    features.append(feature.data(), feature.size());
  };
  if ((eflags & ELF::EF_AMDGPU_FEATURE_XNACK_V4) ==
      ELF::EF_AMDGPU_FEATURE_XNACK_ON_V4)
    addFeature("+xnack");
  if ((eflags & ELF::EF_AMDGPU_FEATURE_SRAMECC_V4) ==
      ELF::EF_AMDGPU_FEATURE_SRAMECC_ON_V4)
    addFeature("+sramecc");
  else if ((eflags & ELF::EF_AMDGPU_FEATURE_SRAMECC_V4) ==
           ELF::EF_AMDGPU_FEATURE_SRAMECC_OFF_V4)
    addFeature("-sramecc");

  if (ctx.arg.osabi != ELF::ELFOSABI_AMDGPU_HSA) {
    Err(ctx) << "AMDGPU object linking is only supported for amdhsa (OSABI "
             << ctx.arg.osabi << ")";
    return;
  }

  std::string error;
  Triple triple(Triple::amdgcn, Triple::NoSubArch, Triple::AMD, Triple::AMDHSA);
  const Target *target = TargetRegistry::lookupTarget(triple, error);
  if (!target) {
    Err(ctx) << "AMDGPU: failed to look up target: " << error;
    return;
  }
  std::unique_ptr<MCSubtargetInfo> sti(
      target->createMCSubtargetInfo(triple, cpu, features));
  if (!sti) {
    Err(ctx) << "AMDGPU: failed to create subtarget info for '" << cpu << "'";
    return;
  }

  bool hasLDS = !ldsSymbols.empty();
  bool hasBarriers = !barriers.empty();
  bool needsReachability =
      (cg.hasLDSUses() || cg.hasBarrierUses()) && !cg.kernels().empty();
  bool needsResource = hasResourceUsage && !cg.kernels().empty();

  // Alias resolution and kernel discovery must be complete before this point:
  // the shared SCC graph is intentionally kernel-reachable and is consumed by
  // both reachability propagation and resource propagation.
  if (needsReachability || needsResource) {
    LLVM_DEBUG(dbgs() << "AMDGPU: building condensed call graph\n");
    if (!cg.buildCondensedGraph(ctx, needsResource))
      return;
  }

  if (needsReachability) {
    LLVM_DEBUG(dbgs() << "AMDGPU: computing kernel reachability\n");
    cg.computeKernelReachability();
  }

  if (hasLDS)
    resolveLDS(ctx, ldsSymbols, cg);

  if (hasBarriers && !cg.kernels().empty())
    resolveNamedBarriers(ctx, barriers, cg, barToUsers);

  if (needsResource) {
    LLVM_DEBUG(dbgs() << "AMDGPU: propagating resource usage\n");
    cg.propagateResourceUsage();
    if (!validateKernelLDSOccupancy(ctx, cg, *sti))
      return;
  }

  LLVM_DEBUG(dbgs() << "AMDGPU: patching kernel descriptors\n");
  patchKernelDescriptors(ctx, cg, sti.get(), hasLDS, hasBarriers);

  LLVM_DEBUG(dbgs() << "AMDGPU: patching HSA metadata\n");
  patchHSAMetadata<ELFT>(ctx, cg, sti.get(), hasLDS);
}

template void elf::resolveAMDGPUObjectLinking<ELF32LE>(Ctx &);
template void elf::resolveAMDGPUObjectLinking<ELF64LE>(Ctx &);
