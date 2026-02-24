#include "arcanum/frontend/SubsetEnforcer.h"

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"

#include <gtest/gtest.h>
#include <string>

namespace arcanum {
namespace {

/// Helper: parse source string into ASTContext and run enforceSubset.
SubsetResult checkSubset(const std::string& code) {
  std::unique_ptr<clang::ASTUnit> ast = clang::tooling::buildASTFromCode(
      code, "test.cpp", std::make_shared<clang::PCHContainerOperations>());
  EXPECT_NE(ast, nullptr);
  return enforceSubset(ast->getASTContext());
}

// ---------------------------------------------------------------------------
// Parameterized: Rejected constructs
// ---------------------------------------------------------------------------

struct RejectedConstructParam {
  const char* name;
  const char* code;
  const char* diagnosticKeyword;
};

class RejectedConstructTest
    : public ::testing::TestWithParam<RejectedConstructParam> {};

struct RejectedConstructName {
  std::string operator()(
      const ::testing::TestParamInfo<RejectedConstructParam>& info) const {
    return info.param.name;
  }
};

TEST_P(RejectedConstructTest, RejectsDisallowedConstruct) {
  auto [name, code, diagnosticKeyword] = GetParam();
  auto result = checkSubset(code);
  EXPECT_FALSE(result.passed) << "Expected rejection for: " << name;
  bool found = false;
  for (const auto& diag : result.diagnostics) {
    if (diag.find(diagnosticKeyword) != std::string::npos) {
      found = true;
    }
  }
  EXPECT_TRUE(found) << "Expected diagnostic containing '" << diagnosticKeyword
                     << "' for: " << name;
}

INSTANTIATE_TEST_SUITE_P(
    SubsetEnforcer, RejectedConstructTest,
    ::testing::Values(RejectedConstructParam{"VirtualFunction",
                                             R"(
    class Base {
    public:
      virtual int foo() { return 0; }
    };
  )",
                                             "virtual"},
                      RejectedConstructParam{"RawPointer",
                                             R"(
    void foo(int* p) {}
  )",
                                             "pointer"},
                      RejectedConstructParam{"NewExpression",
                                             R"(
    void foo() { int* p = new int(42); }
  )",
                                             "new"},
                      RejectedConstructParam{"Throw",
                                             R"(
    void foo() { throw 42; }
  )",
                                             "throw"},
                      RejectedConstructParam{"DoubleType",
                                             R"(
    double foo() { return 3.14; }
  )",
                                             "floating-point"},
                      RejectedConstructParam{"DeleteExpression",
                                             R"(
    void foo() { int* p = new int(42); delete p; }
  )",
                                             "delete"},
                      RejectedConstructParam{"TryCatch",
                                             R"(
    void foo() {
      try { } catch (...) { }
    }
  )",
                                             "try"},
                      RejectedConstructParam{"Goto",
                                             R"(
    void foo() {
      label:
      goto label;
    }
  )",
                                             "goto"},
                      RejectedConstructParam{"TemplateFunction",
                                             R"(
    template<typename T>
    T identity(T x) { return x; }
    void dummy() { identity(42); }
  )",
                                             "template"},
                      RejectedConstructParam{"EarlyReturn",
                                             R"(
    #include <cstdint>
    int32_t foo(int32_t a) {
      if (a < 0) return -1;
      return a;
    }
  )",
                                             "return"},
                      RejectedConstructParam{"Recursion",
                                             R"(
    #include <cstdint>
    int32_t factorial(int32_t n) {
      if (n <= 1) {
        return 1;
      } else {
        return n * factorial(n - 1);
      }
    }
  )",
                                             "recursive"},
                      RejectedConstructParam{"NamespacedFunction",
                                             R"(
    #include <cstdint>
    namespace myns {
      int32_t add(int32_t a, int32_t b) { return a + b; }
    }
  )",
                                             "namespace"},
                      RejectedConstructParam{"ForLoop",
                                             R"(
    #include <cstdint>
    int32_t sum(int32_t n) {
      int32_t s = 0;
      for (int32_t i = 0; i < n; i = i + 1) {
        s = s + i;
      }
      return s;
    }
  )",
                                             "for loop"},
                      RejectedConstructParam{"WhileLoop",
                                             R"(
    #include <cstdint>
    int32_t count(int32_t n) {
      int32_t i = 0;
      while (i < n) {
        i = i + 1;
      }
      return i;
    }
  )",
                                             "while loop"},
                      RejectedConstructParam{"DoWhileLoop",
                                             R"(
    #include <cstdint>
    int32_t count(int32_t n) {
      int32_t i = 0;
      do {
        i = i + 1;
      } while (i < n);
      return i;
    }
  )",
                                             "do-while"},
                      RejectedConstructParam{"SwitchStatement",
                                             R"(
    #include <cstdint>
    int32_t classify(int32_t x) {
      switch (x) {
        case 0: return 0;
        default: return 1;
      }
    }
  )",
                                             "switch"},
                      RejectedConstructParam{"FunctionCalls",
                                             R"(
    #include <cstdint>
    int32_t helper(int32_t x) { return x; }
    int32_t caller(int32_t x) { return helper(x); }
  )",
                                             "function call"},
                      // G6: Range-based for loop rejection
                      RejectedConstructParam{"RangeForLoop",
                                             R"(
    #include <cstdint>
    int32_t sum() {
      int32_t arr[] = {1, 2, 3};
      int32_t s = 0;
      for (int32_t x : arr) {
        s = s + x;
      }
      return s;
    }
  )",
                                             "range-based for"},
                      // G7: Disallowed local variable type
                      RejectedConstructParam{"DisallowedLocalVarType",
                                             R"(
    #include <cstdint>
    int32_t foo(int32_t a) {
      double x = 1.0;
      return a;
    }
  )",
                                             "floating-point"}),
    RejectedConstructName{});

// ---------------------------------------------------------------------------
// Parameterized: Accepted constructs
// ---------------------------------------------------------------------------

struct AcceptedConstructParam {
  const char* name;
  const char* code;
};

class AcceptedConstructTest
    : public ::testing::TestWithParam<AcceptedConstructParam> {};

struct AcceptedConstructName {
  std::string operator()(
      const ::testing::TestParamInfo<AcceptedConstructParam>& info) const {
    return info.param.name;
  }
};

TEST_P(AcceptedConstructTest, AcceptsAllowedConstruct) {
  auto [name, code] = GetParam();
  auto result = checkSubset(code);
  EXPECT_TRUE(result.passed) << "Expected acceptance for: " << name;
  EXPECT_TRUE(result.diagnostics.empty())
      << "Unexpected diagnostics for: " << name;
}

INSTANTIATE_TEST_SUITE_P(
    SubsetEnforcer, AcceptedConstructTest,
    ::testing::Values(AcceptedConstructParam{"Int32ArithmeticFunction",
                                             R"(
    #include <cstdint>
    int32_t add(int32_t a, int32_t b) { return a + b; }
  )"},
                      AcceptedConstructParam{"BoolFunction",
                                             R"(
    #include <cstdint>
    bool isPositive(int32_t x) { return x > 0; }
  )"},
                      AcceptedConstructParam{"IfElse",
                                             R"(
    #include <cstdint>
    int32_t abs(int32_t x) {
      if (x < 0) {
        return -x;
      } else {
        return x;
      }
    }
  )"},
                      AcceptedConstructParam{"AllArithmeticOps",
                                             R"(
    #include <cstdint>
    int32_t compute(int32_t a, int32_t b) {
      int32_t sum = a + b;
      int32_t diff = a - b;
      int32_t prod = a * b;
      int32_t quot = a / b;
      int32_t rem = a % b;
      return sum;
    }
  )"},
                      AcceptedConstructParam{"AllComparisonAndLogicalOps",
                                             R"(
    #include <cstdint>
    bool check(int32_t a, int32_t b) {
      bool r1 = a < b;
      bool r2 = a <= b;
      bool r3 = a > b;
      bool r4 = a >= b;
      bool r5 = a == b;
      bool r6 = a != b;
      bool r7 = r1 && r2;
      bool r8 = r3 || r4;
      bool r9 = !r5;
      return r9;
    }
  )"},
                      AcceptedConstructParam{"VoidReturnFunction",
                                             R"(
    void doNothing() { }
  )"}),
    AcceptedConstructName{});

// ============================================================
// Slice 2: Multi-type support tests
// ============================================================

// [S2] Accept all 8 fixed-width integer types
TEST(SubsetEnforcerTest, AcceptsAllFixedWidthTypes) {
  auto result = checkSubset(R"(
    #include <cstdint>
    int8_t f1(int8_t x) { return x; }
    int16_t f2(int16_t x) { return x; }
    int32_t f3(int32_t x) { return x; }
    int64_t f4(int64_t x) { return x; }
    uint8_t f5(uint8_t x) { return x; }
    uint16_t f6(uint16_t x) { return x; }
    uint32_t f7(uint32_t x) { return x; }
    uint64_t f8(uint64_t x) { return x; }
  )");
  EXPECT_TRUE(result.passed) << "All 8 fixed-width types should be accepted";
  EXPECT_TRUE(result.diagnostics.empty());
}

// [S2] Accept plain 'int' (width 32 on target platforms)
// NOTE: Platform-dependent -- 'int' is accepted because its width (32)
// matches an allowed width on x86_64.  On platforms where 'int' has a
// different width (e.g., 16-bit embedded targets), this test would fail.
// See TODO(SC-1) in SubsetEnforcer.cpp and the corresponding note in
// Lowering.cpp getArcType().
TEST(SubsetEnforcerTest, AcceptsPlainInt) {
  auto result = checkSubset(R"(
    int foo(int a) { return a; }
  )");
  EXPECT_TRUE(result.passed) << "int has width 32; should be accepted";
}

// [S2] Accept 'short' (width 16 on target platforms)
// NOTE: Platform-dependent -- assumes 'short' is 16 bits (standard on
// x86_64).  See TODO(SC-1) for the known spec deviation.
TEST(SubsetEnforcerTest, AcceptsShort) {
  auto result = checkSubset(R"(
    short foo(short a) { return a; }
  )");
  EXPECT_TRUE(result.passed) << "short has width 16; should be accepted";
}

// [S2] Accept 'long' (width 64 on x86_64 Linux)
// NOTE: Platform-dependent -- 'long' is 64 bits on LP64 (Linux x86_64)
// but 32 bits on LLP64 (Windows x64).  This test would behave differently
// on Windows.  See TODO(SC-1) for the known spec deviation.
TEST(SubsetEnforcerTest, AcceptsLong) {
  auto result = checkSubset(R"(
    long foo(long a) { return a; }
  )");
  EXPECT_TRUE(result.passed)
      << "long has width 64 on x86_64; should be accepted";
}

// [S2] Accept 'unsigned int' (width 32 on target platforms)
// NOTE: Platform-dependent -- assumes 'unsigned int' is 32 bits.
// See TODO(SC-1) for the known spec deviation.
TEST(SubsetEnforcerTest, AcceptsUnsignedInt) {
  auto result = checkSubset(R"(
    unsigned int foo(unsigned int a) { return a; }
  )");
  EXPECT_TRUE(result.passed) << "unsigned int has width 32; should be accepted";
}

// [S2] Reject __int128 (unsupported 128-bit width)
TEST(SubsetEnforcerTest, RejectsInt128) {
  auto result = checkSubset(R"(
    __int128 foo(__int128 a) { return a; }
  )");
  EXPECT_FALSE(result.passed);
  bool foundUnsupported = false;
  for (const auto& diag : result.diagnostics) {
    if (diag.find("unsupported width") != std::string::npos) {
      foundUnsupported = true;
    }
  }
  EXPECT_TRUE(foundUnsupported);
}

// ============================================================
// Slice 2: Cast validation tests
// ============================================================

// [S2] Accept static_cast between supported types
TEST(SubsetEnforcerTest, AcceptsStaticCast) {
  auto result = checkSubset(R"(
    #include <cstdint>
    int16_t narrow(int32_t x) {
      return static_cast<int16_t>(x);
    }
  )");
  EXPECT_TRUE(result.passed)
      << "static_cast between supported types should be accepted";
  EXPECT_TRUE(result.diagnostics.empty());
}

// [S2] Reject C-style cast
TEST(SubsetEnforcerTest, RejectsCStyleCast) {
  auto result = checkSubset(R"(
    #include <cstdint>
    int16_t narrow(int32_t x) {
      return (int16_t)x;
    }
  )");
  EXPECT_FALSE(result.passed);
  bool foundCast = false;
  for (const auto& diag : result.diagnostics) {
    if (diag.find("static_cast") != std::string::npos) {
      foundCast = true;
    }
  }
  EXPECT_TRUE(foundCast) << "Should suggest using static_cast";
}

// [S2] Implicit integer conversions should not trigger C-style cast rejection
TEST(SubsetEnforcerTest, ImplicitConversionDoesNotTriggerCastRejection) {
  auto result = checkSubset(R"(
    #include <cstdint>
    int32_t foo(int32_t x) {
      int32_t y = 42;
      return y;
    }
  )");
  EXPECT_TRUE(result.passed);
  for (const auto& diag : result.diagnostics) {
    EXPECT_EQ(diag.find("C-style cast"), std::string::npos)
        << "Implicit conversions should not be flagged as C-style casts";
  }
}

// [S2] Reject reinterpret_cast
TEST(SubsetEnforcerTest, RejectsReinterpretCast) {
  auto result = checkSubset(R"(
    #include <cstdint>
    void foo() {
      int32_t x = 42;
      int64_t y = reinterpret_cast<int64_t>(&x);
    }
  )");
  EXPECT_FALSE(result.passed);
  bool foundCast = false;
  for (const auto& diag : result.diagnostics) {
    if (diag.find("reinterpret_cast") != std::string::npos) {
      foundCast = true;
    }
  }
  EXPECT_TRUE(foundCast);
}

// [S2] Reject const_cast
TEST(SubsetEnforcerTest, RejectsConstCast) {
  auto result = checkSubset(R"(
    #include <cstdint>
    void foo(const int32_t* p) {
      int32_t* q = const_cast<int32_t*>(p);
    }
  )");
  EXPECT_FALSE(result.passed);
  bool foundCast = false;
  for (const auto& diag : result.diagnostics) {
    if (diag.find("const_cast") != std::string::npos) {
      foundCast = true;
    }
  }
  EXPECT_TRUE(foundCast);
}

// [S2] Reject dynamic_cast
TEST(SubsetEnforcerTest, RejectsDynamicCast) {
  auto result = checkSubset(R"(
    class Base {
    public:
      virtual ~Base() = default;
    };
    class Derived : public Base {};
    void foo(Base* b) {
      Derived* d = dynamic_cast<Derived*>(b);
    }
  )");
  EXPECT_FALSE(result.passed);
  bool foundCast = false;
  for (const auto& diag : result.diagnostics) {
    if (diag.find("dynamic_cast") != std::string::npos) {
      foundCast = true;
    }
  }
  EXPECT_TRUE(foundCast);
}

// ============================================================
// Slice 2: Mixed-type binary op tests
// ============================================================

// [S2] Reject mixed-type binary operations
TEST(SubsetEnforcerTest, RejectsMixedTypeBinaryOp) {
  auto result = checkSubset(R"(
    #include <cstdint>
    int32_t foo(int32_t a, int16_t b) {
      return a + static_cast<int32_t>(b);
    }
  )");
  // The static_cast is fine, but a + static_cast<int32_t>(b) should be OK
  // because both sides are int32_t after the cast.
  EXPECT_TRUE(result.passed);
}

// [S2] Reject actually mixed-type arithmetic (different widths)
TEST(SubsetEnforcerTest, RejectsMixedWidthArithmetic) {
  auto result = checkSubset(R"(
    #include <cstdint>
    int32_t foo(int32_t a, int16_t b) {
      return a + b;
    }
  )");
  EXPECT_FALSE(result.passed);
  bool foundMatching = false;
  for (const auto& diag : result.diagnostics) {
    if (diag.find("matching types") != std::string::npos) {
      foundMatching = true;
    }
  }
  EXPECT_TRUE(foundMatching);
}

// [S2] Reject mixed-signedness arithmetic
TEST(SubsetEnforcerTest, RejectsMixedSignednessArithmetic) {
  auto result = checkSubset(R"(
    #include <cstdint>
    int32_t foo(int32_t a, uint32_t b) {
      return a + b;
    }
  )");
  EXPECT_FALSE(result.passed);
  bool foundMatching = false;
  for (const auto& diag : result.diagnostics) {
    if (diag.find("matching types") != std::string::npos) {
      foundMatching = true;
    }
  }
  EXPECT_TRUE(foundMatching);
}

// [S2] Accept same-type binary operations (int16_t + int16_t)
TEST(SubsetEnforcerTest, AcceptsSameTypeBinaryOp) {
  auto result = checkSubset(R"(
    #include <cstdint>
    int16_t add16(int16_t a, int16_t b) {
      return a + b;
    }
  )");
  EXPECT_TRUE(result.passed) << "Same-type binary ops should be accepted";
  EXPECT_TRUE(result.diagnostics.empty());
}

// [S2] Accept same-type unsigned binary operations
TEST(SubsetEnforcerTest, AcceptsSameTypeUnsignedBinaryOp) {
  auto result = checkSubset(R"(
    #include <cstdint>
    uint64_t add64(uint64_t a, uint64_t b) {
      return a + b;
    }
  )");
  EXPECT_TRUE(result.passed);
  EXPECT_TRUE(result.diagnostics.empty());
}

// [S2] Logical ops (&&, ||) on bools should still work
TEST(SubsetEnforcerTest, AcceptsLogicalOpsOnBools) {
  auto result = checkSubset(R"(
    #include <cstdint>
    bool test(int32_t a, int32_t b) {
      bool x = a > 0;
      bool y = b > 0;
      return x && y;
    }
  )");
  EXPECT_TRUE(result.passed);
}

// [S2] Comparison between same-type integers should work
TEST(SubsetEnforcerTest, AcceptsSameTypeComparison) {
  auto result = checkSubset(R"(
    #include <cstdint>
    bool isLess(int16_t a, int16_t b) {
      return a < b;
    }
  )");
  EXPECT_TRUE(result.passed);
}

// [TC-15] Reject static_cast to unsupported type (e.g., float)
TEST(SubsetEnforcerTest, RejectsStaticCastToUnsupportedType) {
  auto result = checkSubset(R"(
    #include <cstdint>
    float narrow(int32_t x) {
      return static_cast<float>(x);
    }
  )");
  EXPECT_FALSE(result.passed);
  bool foundFloat = false;
  for (const auto& diag : result.diagnostics) {
    if (diag.find("floating-point") != std::string::npos) {
      foundFloat = true;
    }
  }
  EXPECT_TRUE(foundFloat) << "Should reject static_cast to float";
}

} // namespace
} // namespace arcanum
