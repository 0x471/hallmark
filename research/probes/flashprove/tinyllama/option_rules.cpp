// The option rules of prover.patch's -without-worst-case, run on the host from the patched sources
// (check 7 of scripts/flashprove_tinyllama.py). Hallmark code, MIT. Built with -DCP_COMMIT (the
// parser of cp_commit, prover/weights_commitment/arguments.h) or -DCP_PROVE (that of cp_prove,
// prover/prover/arguments.h), each with and without -DSERVING_BUILD (cp_prove's only with it: its
// options are tested where -public-weights exists).
//
// Usage: option_rules WEIGHTS_FOLDER CONFIG_FOLDER
// WEIGHTS_FOLDER holds a file *.gguf (cp_commit's parser looks for one); the other paths are not
// read. Each case prints one line; the last line is "ALL AS EXPECTED" or "FAILED", and the exit
// status 0 or 1. Every refusal of the new option has a control that differs only by the option and
// passes, so that a refusal for another reason cannot pass for the rule.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "common/setting.h"
typedef uint32_t u32;
typedef uint64_t u64;
typedef uint8_t u8;
using std::vector;
#if defined(CP_COMMIT)
#include "weights_commitment/arguments.h"
typedef rf::Opt Options;
#elif defined(CP_PROVE)
#include "prover/arguments.h"
typedef Arguments Options;
#else
#error "build with -DCP_COMMIT or -DCP_PROVE"
#endif

static int failures = 0;

// ok: whether parse() must accept; check: -1, or the value the field must have after acceptance.
static void expect(const char *name, std::vector<std::string> args, bool ok, int check) {
  std::vector<char *> argv{(char *)"program"};
  for (std::string &s : args) argv.push_back(&s[0]);
  Options o;
  const bool got = o.parse((int)argv.size(), argv.data());
#if defined(CP_COMMIT)
  const int value = (int)o.worst_case;
  const char *field = "worst_case";
#else
  const int value = (int)o.without_worst_case;
  const char *field = "without_worst_case";
#endif
  const bool good = got == ok && (!got || check < 0 || value == check);
  printf("case %s: parse %s, %s %d: %s\n", name, got ? "accepts" : "refuses", field, got ? value : -1,
         good ? "as expected" : "NOT AS EXPECTED");
  failures += !good;
}

int main(int argc, char **argv) {
  if (argc != 3) return fprintf(stderr, "usage: option_rules WEIGHTS_FOLDER CONFIG_FOLDER\n"), 2;
  const std::string w = argv[1], c = argv[2];
#if defined(CP_COMMIT)
  const std::vector<std::string> base = {"-weights", w, "-config", c, "-M", "1", "-ivk", "/nonexistent/i", "-isk", "/nonexistent/k"};
  auto with = [&](std::vector<std::string> extra) {
    std::vector<std::string> v = base;
    v.insert(v.end(), extra.begin(), extra.end());
    return v;
  };
  const std::vector<std::string> second = {"-ivk", "/nonexistent/i", "-isk", "/nonexistent/k", "-M", "2"};
  std::vector<std::string> second_with = second;
  second_with.push_back("-without-worst-case");
#ifdef SERVING_BUILD
  expect("model files: the check runs (upstream)", with({}), true, 1);
  expect("-public-weights: the check runs (upstream)", with({"-public-weights", "/nonexistent/p"}), true, 1);
  expect("-public-weights -without-worst-case: the check is left out", with({"-public-weights", "/nonexistent/p", "-without-worst-case"}), true, 0);
  expect("-without-worst-case without -public-weights", with({"-without-worst-case"}), false, -1);
  expect("control: -public-weights -worst-case-table", with({"-public-weights", "/nonexistent/p", "-worst-case-table", "/nonexistent/t"}), true, 1);
  expect("-without-worst-case with -worst-case-table", with({"-public-weights", "/nonexistent/p", "-without-worst-case", "-worst-case-table", "/nonexistent/t"}), false, -1);
  expect("control: -public-weights -worst-case", with({"-public-weights", "/nonexistent/p", "-worst-case"}), true, 1);
  expect("-without-worst-case with -worst-case", with({"-public-weights", "/nonexistent/p", "-without-worst-case", "-worst-case"}), false, -1);
  expect("control: the second form", second, true, -1);
  expect("-without-worst-case in the second form", second_with, false, -1);
#else
  expect("model files: the check runs (upstream)", with({}), true, 1);
  expect("-test-seed: the check is left out (upstream)", with({"-test-seed", "/nonexistent/s"}), true, 0);
  expect("-without-worst-case in a test build (no -public-weights there)", with({"-test-seed", "/nonexistent/s", "-without-worst-case"}), false, -1);
#endif
#else
  const std::vector<std::string> one = {"-isk", "/nonexistent/k", "-in", "/nonexistent/i", "-out", "/nonexistent/o"};
  auto with = [&](std::vector<std::string> extra) {
    std::vector<std::string> v = one;
    v.insert(v.end(), extra.begin(), extra.end());
    return v;
  };
  (void)w, (void)c;
  expect("control: neither option", with({}), true, 0);
  expect("control: -public-weights", with({"-public-weights", "/nonexistent/p"}), true, 0);
  expect("-public-weights -without-worst-case", with({"-public-weights", "/nonexistent/p", "-without-worst-case"}), true, 1);
  expect("-without-worst-case without -public-weights", with({"-without-worst-case"}), false, -1);
#endif
  printf("%s\n", failures ? "FAILED" : "ALL AS EXPECTED");
  return failures != 0;
}
