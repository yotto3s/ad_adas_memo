#include "frontend/SubsetEnforcer.h"

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

namespace arcanum {
namespace {

/// Helper: parse C++ source and run SubsetEnforcer, return result.
SubsetEnforcerResult enforceOnSource(const std::string &source) {
    auto ast = clang::tooling::buildASTFromCodeWithArgs(
        source, {"-std=c++20", "-fsyntax-only"}, "test_input.cpp");
    EXPECT_NE(ast, nullptr);
    if (!ast)
        return SubsetEnforcerResult{};

    SubsetEnforcer enforcer(ast->getASTContext());
    return enforcer.enforce();
}

// ============================================================
// Accepted code: should pass with no violations
// ============================================================

TEST(SubsetEnforcerTest, AcceptsSimpleFunction) {
    auto result = enforceOnSource(R"cpp(
        #include <cstdint>
        int32_t add(int32_t a, int32_t b) {
            return a + b;
        }
    )cpp");
    EXPECT_TRUE(result.passed());
}

TEST(SubsetEnforcerTest, AcceptsIfElse) {
    auto result = enforceOnSource(R"cpp(
        #include <cstdint>
        int32_t max_val(int32_t a, int32_t b) {
            if (a > b) {
                return a;
            } else {
                return b;
            }
        }
    )cpp");
    EXPECT_TRUE(result.passed());
}

TEST(SubsetEnforcerTest, AcceptsStaticCast) {
    auto result = enforceOnSource(R"cpp(
        #include <cstdint>
        int32_t convert(int64_t x) {
            return static_cast<int32_t>(x);
        }
    )cpp");
    EXPECT_TRUE(result.passed());
}

TEST(SubsetEnforcerTest, AcceptsBoolOperations) {
    auto result = enforceOnSource(R"cpp(
        bool check(int a, int b) {
            return a > 0 && b > 0;
        }
    )cpp");
    EXPECT_TRUE(result.passed());
}

TEST(SubsetEnforcerTest, AcceptsConstexprFunction) {
    auto result = enforceOnSource(R"cpp(
        constexpr int square(int x) {
            return x * x;
        }
    )cpp");
    EXPECT_TRUE(result.passed());
}

TEST(SubsetEnforcerTest, AcceptsReferences) {
    auto result = enforceOnSource(R"cpp(
        void increment(int& x) {
            x = x + 1;
        }
    )cpp");
    EXPECT_TRUE(result.passed());
}

TEST(SubsetEnforcerTest, AcceptsConstReferences) {
    auto result = enforceOnSource(R"cpp(
        int read_val(const int& x) {
            return x;
        }
    )cpp");
    EXPECT_TRUE(result.passed());
}

// ============================================================
// Rejected code: should produce violations
// ============================================================

TEST(SubsetEnforcerTest, RejectsVirtualFunction) {
    auto result = enforceOnSource(R"cpp(
        class Base {
        public:
            virtual int compute() { return 0; }
        };
    )cpp");
    EXPECT_FALSE(result.passed());
    ASSERT_GE(result.violations.size(), 1u);
    EXPECT_NE(result.violations[0].description.find("virtual"),
              std::string::npos);
}

TEST(SubsetEnforcerTest, RejectsThrow) {
    auto result = enforceOnSource(R"cpp(
        #include <stdexcept>
        void fail() {
            throw std::runtime_error("error");
        }
    )cpp");
    EXPECT_FALSE(result.passed());
    ASSERT_GE(result.violations.size(), 1u);
    EXPECT_NE(result.violations[0].description.find("throw"),
              std::string::npos);
}

TEST(SubsetEnforcerTest, RejectsTryCatch) {
    auto result = enforceOnSource(R"cpp(
        void attempt() {
            try {
                int x = 1;
            } catch (...) {
            }
        }
    )cpp");
    EXPECT_FALSE(result.passed());
    ASSERT_GE(result.violations.size(), 1u);
    EXPECT_NE(result.violations[0].description.find("try"),
              std::string::npos);
}

TEST(SubsetEnforcerTest, RejectsNew) {
    auto result = enforceOnSource(R"cpp(
        void allocate() {
            int* p = new int(42);
        }
    )cpp");
    EXPECT_FALSE(result.passed());
    // Should have violations for both new and the raw pointer variable
    EXPECT_GE(result.violations.size(), 1u);
}

TEST(SubsetEnforcerTest, RejectsDelete) {
    auto result = enforceOnSource(R"cpp(
        void deallocate(int* p) {
            delete p;
        }
    )cpp");
    EXPECT_FALSE(result.passed());
    EXPECT_GE(result.violations.size(), 1u);
}

TEST(SubsetEnforcerTest, RejectsGoto) {
    auto result = enforceOnSource(R"cpp(
        void jump() {
            goto end;
            int x = 1;
            end:
            return;
        }
    )cpp");
    EXPECT_FALSE(result.passed());
    ASSERT_GE(result.violations.size(), 1u);
    EXPECT_NE(result.violations[0].description.find("goto"),
              std::string::npos);
}

TEST(SubsetEnforcerTest, RejectsReinterpretCast) {
    auto result = enforceOnSource(R"cpp(
        void cast_bad(int x) {
            auto* p = reinterpret_cast<float*>(&x);
        }
    )cpp");
    EXPECT_FALSE(result.passed());
    EXPECT_GE(result.violations.size(), 1u);
}

TEST(SubsetEnforcerTest, RejectsDynamicCast) {
    auto result = enforceOnSource(R"cpp(
        class Base { public: virtual ~Base() {} };
        class Derived : public Base {};
        void cast_bad(Base* b) {
            auto* d = dynamic_cast<Derived*>(b);
        }
    )cpp");
    EXPECT_FALSE(result.passed());
    EXPECT_GE(result.violations.size(), 1u);
}

TEST(SubsetEnforcerTest, RejectsCStyleCast) {
    auto result = enforceOnSource(R"cpp(
        void cast_bad() {
            double d = 3.14;
            int x = (int)d;
        }
    )cpp");
    EXPECT_FALSE(result.passed());
    ASSERT_GE(result.violations.size(), 1u);
}

TEST(SubsetEnforcerTest, RejectsRawPointerVariable) {
    auto result = enforceOnSource(R"cpp(
        void use_ptr() {
            int x = 10;
            int* p = &x;
        }
    )cpp");
    EXPECT_FALSE(result.passed());
    ASSERT_GE(result.violations.size(), 1u);
    EXPECT_NE(result.violations[0].description.find("raw pointer"),
              std::string::npos);
}

TEST(SubsetEnforcerTest, RejectsRawPointerParameter) {
    auto result = enforceOnSource(R"cpp(
        void process(int* data, int size) {}
    )cpp");
    EXPECT_FALSE(result.passed());
    ASSERT_GE(result.violations.size(), 1u);
    EXPECT_NE(result.violations[0].description.find("raw pointer"),
              std::string::npos);
}

// ============================================================
// Multiple violations
// ============================================================

TEST(SubsetEnforcerTest, ReportsMultipleViolations) {
    auto result = enforceOnSource(R"cpp(
        class Sensor {
        public:
            virtual int read() { return 0; }
        };

        void process(int* data) {
            goto end;
            end:
            return;
        }
    )cpp");
    EXPECT_FALSE(result.passed());
    // virtual + raw pointer param + goto = at least 3
    EXPECT_GE(result.violations.size(), 3u);
}

// ============================================================
// Error count
// ============================================================

TEST(SubsetEnforcerTest, ErrorCountMatchesErrors) {
    auto result = enforceOnSource(R"cpp(
        void bad() {
            goto end;
            end:
            return;
        }
    )cpp");
    EXPECT_EQ(result.errorCount(), result.violations.size());
}

} // namespace
} // namespace arcanum
