#ifndef ARCANUM_FRONTEND_TESTS_TESTHELPERS_H
#define ARCANUM_FRONTEND_TESTS_TESTHELPERS_H

#include "arcanum/frontend/ContractParser.h"

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"

#include <gtest/gtest.h>

namespace arcanum {
namespace testing {

/// Shared helper: parse a C++ source string and return the contract map.
/// The ASTUnit is returned via `astOut` to keep the AST alive for inspection.
inline std::map<const clang::FunctionDecl*, ContractInfo>
parseFromSource(const std::string& code,
                std::unique_ptr<clang::ASTUnit>& astOut) {
  astOut = clang::tooling::buildASTFromCodeWithArgs(
      code, {"-fparse-all-comments"}, "test.cpp", "arcanum-test",
      std::make_shared<clang::PCHContainerOperations>());
  EXPECT_NE(astOut, nullptr);
  return parseContracts(astOut->getASTContext());
}

} // namespace testing
} // namespace arcanum

#endif // ARCANUM_FRONTEND_TESTS_TESTHELPERS_H
