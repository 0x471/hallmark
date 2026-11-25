// Runs on a CPU the part of cp_commit's and cp_prove's set-up that needs only a kit (policy.txt and
// KL.bin), with upstream's code: the parameter-file reader (prover/prover/policy.h), the version
// word (s8::set), the weight table (W's rows by layoutW(), prover/common/weight_layout.h, then
// HfParams::from_policy(), prover/hash_commitment/hf_opening_params.h, as HfShape::from_policy()
// calls them for both programs), the plan of attention of each layer group (APlan::build(),
// prover/attn/plan.cuh) and the layer layout of each group (make_layer_layout(),
// prover/prover/layout.h). They are filled as hf_shape.cuh, read_inputs()
// (prover/prover/inputs.cuh), other_setup.cuh and layer_setup.cuh fill them. A few checks that live
// in CUDA files are copied here, each marked with its source. A refusal prints upstream's message
// and ends with status 2; an admitted kit ends with the line "prover set-up: admitted ..." and
// status 0.
// The layers' scale words are held to layer_words_fit() (prover/prover/forward.cuh), which
// read_inputs() calls on the host for every layer, and attention to two rules of AForkHid
// (prover/attn/fork.cuh): the column lengths of its five committed matrices, and lt and lg.
// Not run here, although each needs only the kit: the rest of AForkHid::setup() (5 <= lmbA <= kT + 5,
// the pad tree's size), AFront::plan()'s limit on a lookup call's pieces of one segment kind
// (prover/attn/front.cuh) and the output head's set-up (prover/head/). A kit that one of these
// refuses is admitted here and stops at cp_prove's start.
// Not run here either: anything on a GPU or a commitment, the memory that a table of this size
// takes, and the hiding conditions, which are only printed (a commitment made with -test-seed is not
// held to them).
// usage: prover_setup <kit folder>
// Build (from the patched tree's prover/ folder): c++ -std=c++17 -O2 -DBLK_FIELD=2 -DBLK_HASH=1
//   -I . -I ../blocks prover_setup.cpp
#include "common/field_host.h" // u32 and u64, as the prover's build has them before plan.cuh
#include "prover/policy.h"
#include "attn/plan.cuh"
#include "frame/hiding_conditions.h"
#include "prover/layout.h"
#include "hash_commitment/hf_opening_params.h"
#include <algorithm>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

// weight_layout.h asks its includer for these names. Only layoutW()'s row counts are used here, so
// the field's elements are empty.
namespace wrows {
using std::vector;
static const int SW = 256;
struct E {};
static const E E0{}, E1{};
[[maybe_unused]] static E ebase(u64) { return E(); }
[[maybe_unused]] static E esub(E, E) { return E(); }
[[maybe_unused]] static E emul(E, E) { return E(); }
#include "common/weight_layout.h"
} // namespace wrows

static int log2_ceil(size_t x) { // l2i() of prover/hiding/tail.cuh
  int l = 0;
  while (((size_t)1 << l) < x) l++;
  return l;
}

// Copied from prover/prover/forward.cuh, a CUDA file: LayerScaleSet, norm_words_fit() and
// layer_words_fit(), which read_inputs() (prover/prover/inputs.cuh) calls on the host.
struct LayerScaleSet {
  int kq, kk, kv, ko, kg, ku, kd, M1, K1, M2, K2, kr, kgR, kuR, rp = 1, sh = 8; // M1: first norm
};
static inline bool norm_words_fit(int M, int K, int xl) {
  return M >= 24 && M <= 47 && K >= 17 && K <= 32 && K >= M - (xl == 5 && K <= 23 ? 13 : 14);
}
static inline bool layer_words_fit(const LayerScaleSet &w, int l, const LayerShape &s) {
  auto shift = [&](const char *n, int k, int form) {
    const int hi = form == 0 ? 16 : form == 1 ? 24 : 32;
    if (k >= 1 && k <= hi) return true;
    printf("layer constants: layer %d: shift %s = %d is outside the range for digit form %d, which is 1 to %d\n", l, n,
           k, form, hi);
    return false;
  };
  auto norm = [&](int n, int M, int K) {
    if (norm_words_fit(M, K, s.xl)) return true;
    printf("layer constants: layer %d: M%d and K%d refused; they must satisfy 24 <= M%d <= 47, 17 <= K%d <= 32 and K%d "
           ">= M%d - 14 (M%d - 13 with five limbs and K%d <= 23)\n",
           l, n, n, n, n, n, n, n, n);
    return false;
  };
  if (!shift("kg", w.kg, s.hg) || !shift("ku", w.ku, s.hu) || !shift("kd", w.kd, s.hdn) || !norm(2, w.M2, w.K2))
    return false;
  if (s.prof >= 2) {
    if (!shift("kq", w.kq, s.hqk) || !shift("kk", w.kk, s.hqk) || !shift("kv", w.kv, s.hv) || !shift("ko", w.ko, s.ho))
      return false;
    if (w.kq != w.kk && (w.kq < 16 || w.kk < 16)) {
      printf("layer constants: layer %d: kq and kk must be equal when either of them is below 16\n", l);
      return false;
    }
    if (!norm(1, w.M1, w.K1)) return false;
  }
  return !moe_at_layer(s.prof, l) || (shift("kr", w.kr, s.hr) && shift("kgR", w.kgR, s.hg) && shift("kuR", w.kuR, s.hu));
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("usage: prover_setup <kit folder>\n");
    return 2;
  }
  const std::string kit = argv[1];
  dup2(1, 2); // upstream's refusals on stderr go where this program's lines go
  setvbuf(stdout, nullptr, _IONBF, 0);
  Policy policy;
  if (!policy.read(kit + "/policy.txt")) return 2;
  if (const std::string why = s8::set(policy.frame); !why.empty()) {
    printf("policy: %s\n", why.c_str());
    return 2;
  }
  const std::string missed = blk::hiding_conditions_missed_by_the_file(policy.frame);
  printf("hiding conditions that follow from the file alone: %s\n", missed.empty() ? "met" : missed.c_str());
  const int lt = policy["lt"], Lr = policy["L"], ll = policy["ll"], d = policy["d"], f = policy["f"];
  const int la = policy["la"], nmin = policy["nmin"], D = policy["D"], H = policy["H"], Gk = policy["G"];
  const int dh = policy["dh"], NE = policy["NE"], fe = policy["fe"], xl = (int)policy.value("xl");
  const int Lam = 1 << ll, NGr = Lr / Lam, ldh = log2_ceil(dh), mo = NE > 0, KLW = 13;
  { // The weight table: HfShape::from_policy(), prover/hash_commitment/hf_shape.cuh.
    const int prof = policy["prof"], eh = prof >> 4, lk = (int)policy.frame.block_log;
    if (lk < ELKW_LEAST || lk > ELKW_MOST) {
      printf("the last number of the parameter file's line table must be from %d to %d\n", ELKW_LEAST, ELKW_MOST);
      return 2;
    }
    wrows::ShapeW sh;
    sh.L = Lr, sh.d = d, sh.dh = dh, sh.dkv = (size_t)Gk * dh, sh.dq = (size_t)H * dh, sh.f = f, sh.fe = fe, sh.NE = NE;
    sh.V = eh ? policy["V"] : 0;
    sh.moeMode = ((prof & 15) >= 3 && NE > 0) ? 1 : 0;
    if (eh && (wrows::clgW(sh.V) < 1 || wrows::clgW(sh.V) > lk)) {
      printf("the last number of the parameter file's line table is too small, or V is below 2 (w-list-token-factor)\n");
      return 2;
    }
    const wrows::LayW Y = wrows::layoutW(sh, lk, D, 0, eh ? 1 : 0);
    if (Y.m0 > ((size_t)1 << ELMW)) {
      printf("the model is too large for this parameter file: W would have more than 2^%d rows\n", ELMW);
      return 2;
    }
    hf::HfParams P;
    if (P.from_policy(policy.frame, Y.m0)) return 2; // it has printed why
    P.print();
    printf("weight table: admitted (W: %zu rows of 2^%d columns in %zu units; 2^%d long rows of %llu data entries "
           "in %llu, %llu masks, code length 2^%d; a tree's stored level at most %d)\n",
           Y.m0, lk, Y.U.size(), P.b, (unsigned long long)P.l, (unsigned long long)P.i0, (unsigned long long)P.nm, P.ln,
           std::min(P.lcl, P.keep_level_most()));
  }
  identity_cap() = (int)policy.value("lidb");
  if (identity_cap() > EIDB) { // read_inputs()
    printf("policy: plan-identity-cap: lidb %d is above this build's EIDB %d\n", identity_cap(), EIDB);
    return 2;
  }
  std::vector<u32> KL;
  {
    size_t words = (size_t)KLW * Lr;
    for (int l = 0; l < Lr; l++) words += moe_at_layer(mo ? 3 : 2, l) ? 3 : 0;
    KL.assign(words, 0);
    FILE *kf = fopen((kit + "/KL.bin").c_str(), "rb");
    const bool whole = kf && fread(KL.data(), 4, KL.size(), kf) == KL.size() && fgetc(kf) == EOF;
    if (kf) fclose(kf);
    if (!whole) { // read_inputs()
      printf("layer constants: KL.bin is not 13 words per layer plus 3 per layer with experts\n");
      return 2;
    }
    for (int l = 0; l < Lr; l++) // read_inputs()
      if (KL[(size_t)KLW * l + 11] < 4 || KL[(size_t)KLW * l + 11] > 12) {
        printf("layer constants: af-policy-sh: layer %d: word 11, attention's shift, must be from 4 to 12\n", l);
        return 2;
      }
  }
  { // The layers' scale words: read_inputs(), with Prover::layer_constants() and digit_forms().
    LayerShape s;
    s.prof = mo ? 3 : 2;
    s.xl = xl;
    s.hqk = policy["hq"];
    s.hv = policy["hv"];
    s.ho = policy["ho"];
    s.hg = policy["hgD"];
    s.hu = policy["huD"];
    s.hdn = policy["hdD"];
    s.hr = mo ? (int)policy["hr"] : 0;
    for (int l = 0; l < Lr; l++) {
      const u32 none[3] = {0, 0, 0}, *w = &KL[(size_t)KLW * l];
      const u32 *e = (mo && (l & 1)) ? &KL[(size_t)KLW * Lr + 3 * (l / 2)] : none;
      const LayerScaleSet words{(int)w[0], (int)w[1], (int)w[2],  (int)w[3], (int)w[4], (int)w[5], (int)w[6],  (int)w[7],
                                (int)w[8], (int)w[9], (int)w[10], (int)e[0], (int)e[1], (int)e[2], (int)w[12], (int)w[11]};
      if (!layer_words_fit(words, l, s)) return 2;
    }
  }
  // Attention: one plan for each distinct list of the groups' shifts (other_setup.cuh).
  std::map<std::vector<int>, int> plans;
  for (int g = 0; g < NGr; g++) {
    std::vector<int> sh;
    for (int lam = 0; lam < Lam; lam++) sh.push_back((int)KL[(size_t)KLW * (g * Lam + lam) + 11]);
    if (plans.count(sh)) continue;
    APlan P;
    AP &p = P.p;
    p.lt = lt;
    p.lB = policy["lB"];
    p.H = H;
    p.lgk = log2_ceil(Gk);
    p.ldh = ldh;
    p.ll = ll;
    p.S = policy["S"];
    p.rx = policy["rx"];
    p.kFM = policy["kFM"];
    p.kFD = policy["kFD"];
    p.D = D;
    p.lmS = policy["lmS"];
    p.lwH = (int)policy["lmh"] - lt;
    p.lwO = (int)policy["lmo"] - lt;
    p.lwQ = (int)policy["lmq"] - lt;
    p.la = la;
    p.lmb = policy["lmbA"];
    p.nmin = nmin;
    p.iota = policy["iotaA_0"];
    P.sh = sh;
    P.form = 3;
    if (p.D != 2048) { // AForkHid::setup(), prover/attn/fork.cuh
      printf("attention: this prover supports only the ring degree D = 2,048\n");
      return 2;
    }
    P.build();
    int shm = 0; // a_forward_ok(), prover/attn/forward.cuh
    for (int s : P.sh) shm = std::max(shm, s);
    if (!((p.dh == 32 || p.dh == 64 || p.dh == 128) && p.lt >= 6 && p.lt <= 15 && p.lB >= 4 && p.S <= 14 &&
          4 * p.Smax + ((I64)p.u0 << shm) < ((I64)1 << 32) && p.Lam <= 16)) {
      printf("attention: the forward pass does not support this shape (a_forward_ok)\n");
      return 2;
    }
    if (P.dT < std::max(p.lmS, p.lt + p.ldh)) { // AFront::plan(), prover/attn/front.cuh
      printf("policy: forest-depth-A: a tree has 2^%d keys, fewer than 2^%d\n", P.dT, std::max(p.lmS, p.lt + p.ldh));
      return 2;
    }
    if (p.iota != 32 + P.dT) { // AFront::plan()
      printf("policy: policy-iotaA: the line iotaA_0 must be 32 + %d; it is %d\n", P.dT, p.iota);
      return 2;
    }
    { // AForkHid::layoutMatrices(), prover/attn/fork.cuh: each committed matrix has whole columns of D.
      const size_t mm[5] = {(size_t)1 << p.lmS, (size_t)p.T << p.lwH, (size_t)p.T << p.lwO, (size_t)1 << p.lmb,
                            (size_t)p.T << p.lwQ};
      for (size_t m : mm)
        if (m % (size_t)p.D) {
          printf("attention: the column length of a committed matrix (2^lmS, 2^lmh, 2^lmo, 2^lmbA or 2^lmq) is not a "
                 "multiple of the ring degree D\n");
          return 2;
        }
    }
    if (p.lt > 14 || 2 * p.lt + p.lgk > 32) { // AForkHid::countSlots()
      printf("attention: this prover needs lt <= 14 (at most 2^14 tokens) and 2 lt + lg <= 32, where 2^lg is the "
             "number of key-value heads; this parameter file has lt = %d and lg = %d\n",
             p.lt, p.lgk);
      return 2;
    }
    plans[sh] = P.dT;
  }
  // Layers: one layout per group (layer_setup.cuh, Prover::g_setup_layout()).
  std::string lists;
  for (auto &c : policy.cl) {
    lists += c.name + ":";
    for (size_t i = 0; i < c.seg.size(); i++) lists += (i ? "," : "") + c.seg[i];
    lists += ";";
  }
  lists.pop_back();
  for (int g = 0; g < NGr; g++) {
    LayerShape S{lt,
                 ll,
                 d,
                 H,
                 Gk,
                 dh,
                 f,
                 fe,
                 mo ? NE : 0,
                 {(int)KL[7], (int)KL[9]},
                 {(int)KL[8], (int)KL[10]},
                 {(long long)(policy["eps1_lo"] | ((u64)policy["eps1_hi"] << 32)),
                  (long long)(policy["eps2_lo"] | ((u64)policy["eps2_hi"] << 32))},
                 g,
                 mo ? 3 : 2,
                 nmin,
                 la};
    S.hqk = policy["hq"]; // Prover::digit_forms(), prover/prover/prover.h
    S.hv = policy["hv"];
    S.ho = policy["ho"];
    S.hg = policy["hgD"];
    S.hu = policy["huD"];
    S.hdn = policy["hdD"];
    S.hr = mo ? (int)policy["hr"] : 0;
    S.rcl = policy.m.count("rcl") ? 1 : 0;
    S.gcl = policy.m.count("gcl") ? 1 : 0;
    S.xl = xl;
    S.att = 1;
    S.lists = lists.c_str();
    make_layer_layout(S);
  }
  printf("prover set-up: admitted (%d layer groups, %zu attention plan%s, kFM %u, kFD %u)\n", NGr, plans.size(),
         plans.size() == 1 ? "" : "s", policy["kFM"], policy["kFD"]);
  return 0;
}
