#include "frontend/SubsetEnforcer.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceManager.h"

#include <algorithm>
#include <string>

namespace arcanum {
namespace {

/// Bit width of int32_t, used for type-width checks.
constexpr uint64_t INT32_BIT_WIDTH = 32;

class SubsetVisitor : public clang::RecursiveASTVisitor<SubsetVisitor> {
public:
  explicit SubsetVisitor(clang::ASTContext& ctx, SubsetResult& result)
      : ctx(ctx), result(result) {}

  bool VisitFunctionDecl(clang::FunctionDecl* decl) {
    if (!decl->hasBody()) {
      return true; // Skip declarations without bodies
    }
    // Skip compiler-generated functions
    if (decl->isImplicit()) {
      return true;
    }
    // Reject user-defined functions inside namespaces or classes.  Slice 1
    // only processes top-level declarations; namespaced functions would pass
    // SubsetEnforcer but be silently skipped by ContractParser/Lowering.
    // We skip functions from system headers (e.g., std:: from <cstdint>).
    if (auto* dc = decl->getDeclContext()) {
      if (!dc->isTranslationUnit()) {
        bool isSystemHeader =
            ctx.getSourceManager().isInSystemHeader(decl->getLocation());
        // Allow methods (caught by virtual/class checks) and system headers
        if (!llvm::isa<clang::CXXMethodDecl>(decl) && !isSystemHeader) {
          addDiagnostic(
              decl->getLocation(),
              "functions inside namespaces or classes are not allowed in "
              "Slice 1 (only top-level functions are supported)");
          return true;
        }
      }
    }
    // Reject virtual functions
    if (auto* method = llvm::dyn_cast<clang::CXXMethodDecl>(decl)) {
      if (method->isVirtual()) {
        addDiagnostic(decl->getLocation(), "virtual functions are not allowed");
        return true;
      }
    }
    // Reject templates
    if (decl->isTemplated()) {
      addDiagnostic(decl->getLocation(),
                    "template functions are not allowed in Slice 1");
      return true;
    }
    // Check single return: reject early returns (return followed by more
    // statements in the same block).  Returns in terminal if/else branches
    // are allowed since they form a single structured exit.
    if (decl->hasBody()) {
      if (hasEarlyReturn(decl->getBody())) {
        addDiagnostic(decl->getLocation(),
                      "early return statements are not allowed in Slice 1");
      }
    }
    // Check non-recursive (function does not call itself)
    if (decl->hasBody()) {
      if (callsSelf(decl, decl->getBody())) {
        addDiagnostic(decl->getLocation(),
                      "recursive functions are not allowed in Slice 1");
      }
    }
    // Check return type
    checkType(decl->getReturnType(), decl->getLocation());
    // Check parameter types
    for (const auto* param : decl->parameters()) {
      checkType(param->getType(), param->getLocation());
    }
    return true;
  }

  bool VisitVarDecl(clang::VarDecl* decl) {
    if (decl->isImplicit()) {
      return true;
    }
    checkType(decl->getType(), decl->getLocation());
    return true;
  }

  bool VisitCXXNewExpr(clang::CXXNewExpr* expr) {
    addDiagnostic(expr->getBeginLoc(),
                  "dynamic allocation (new) is not allowed");
    return true;
  }

  bool VisitCXXDeleteExpr(clang::CXXDeleteExpr* expr) {
    addDiagnostic(expr->getBeginLoc(),
                  "dynamic deallocation (delete) is not allowed");
    return true;
  }

  bool VisitCXXThrowExpr(clang::CXXThrowExpr* expr) {
    addDiagnostic(expr->getBeginLoc(), "throw expressions are not allowed");
    return true;
  }

  bool VisitCXXTryStmt(clang::CXXTryStmt* stmt) {
    addDiagnostic(stmt->getBeginLoc(), "try/catch is not allowed");
    return true;
  }

  bool VisitGotoStmt(clang::GotoStmt* stmt) {
    addDiagnostic(stmt->getBeginLoc(), "goto is not allowed");
    return true;
  }

private:
  /// Check whether a statement (or any nested statement) contains a return.
  bool containsReturn(const clang::Stmt* stmt) {
    if (stmt == nullptr) {
      return false;
}
    if (llvm::isa<clang::ReturnStmt>(stmt)) {
      return true;
}
    return std::any_of(stmt->child_begin(), stmt->child_end(),
                       [this](const clang::Stmt* child) {
                         return containsReturn(child);
                       });
  }

  /// Check for early returns: a return is "early" if it can cause the
  /// function to exit before reaching the end of a compound statement.
  /// Returns inside terminal if/else branches (where both branches return)
  /// are fine because they represent structured single-exit.
  bool hasEarlyReturn(const clang::Stmt* stmt) {
    if (stmt == nullptr) {
      return false;
}
    if (const auto* compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
      for (const auto *it = compound->body_begin(); it != compound->body_end(); ++it) {
        bool isLast = (std::next(it) == compound->body_end());
        // A bare return statement followed by more statements
        if (llvm::isa<clang::ReturnStmt>(*it) && !isLast) {
          return true;
        }
        // An if-without-else that contains a return, followed by more
        // statements -- this is the "guard clause" / early return pattern.
        if (!isLast) {
          if (const auto* ifStmt = llvm::dyn_cast<clang::IfStmt>(*it)) {
            if ((ifStmt->getElse() == nullptr) && containsReturn(ifStmt->getThen())) {
              return true;
            }
          }
        }
        if (hasEarlyReturn(*it)) {
          return true;
        }
      }
    } else {
      for (const auto* child : stmt->children()) {
        if (hasEarlyReturn(child)) {
          return true;
}
      }
    }
    return false;
  }

  bool callsSelf(const clang::FunctionDecl* funcDecl, const clang::Stmt* stmt) {
    if (stmt == nullptr) {
      return false;
}
    if (const auto* call = llvm::dyn_cast<clang::CallExpr>(stmt)) {
      if (const auto* callee = call->getDirectCallee()) {
        if (callee->getCanonicalDecl() == funcDecl->getCanonicalDecl()) {
          return true;
        }
      }
    }
    return std::any_of(stmt->child_begin(), stmt->child_end(),
                       [this, funcDecl](const clang::Stmt* child) {
                         return callsSelf(funcDecl, child);
                       });
  }

  void checkType(clang::QualType type, clang::SourceLocation loc) {
    type = type.getCanonicalType();
    // Allow void (for functions returning void, though Slice 1 expects
    // int32_t/bool)
    if (type->isVoidType()) {
      return;
    }
    // Allow bool
    if (type->isBooleanType()) {
      return;
    }
    // Allow int32_t (which is a typedef for int on most platforms, but check
    // for 32-bit signed integer)
    if (const auto* bt = type->getAs<clang::BuiltinType>()) {
      if (bt->getKind() == clang::BuiltinType::Int) {
        return; // int32_t maps to int on most platforms
      }
    }
    // Check for typedef to int32_t specifically
    if (type->isIntegerType()) {
      auto width = ctx.getTypeSize(type);
      if (width == INT32_BIT_WIDTH && type->isSignedIntegerType()) {
        return;
      }
    }
    // Reject everything else
    if (type->isPointerType()) {
      addDiagnostic(loc, "raw pointer types are not allowed");
    } else if (type->isFloatingType()) {
      addDiagnostic(loc, "floating-point types are not allowed in Slice 1");
    } else {
      addDiagnostic(loc,
                    "type '" + type.getAsString() +
                        "' is not allowed in Slice 1 (only int32_t and bool)");
    }
  }

  void addDiagnostic(clang::SourceLocation loc, const std::string& msg) {
    result.passed = false;
    auto& sm = ctx.getSourceManager();
    if (loc.isValid()) {
      auto presumed = sm.getPresumedLoc(loc);
      if (presumed.isValid()) {
        result.diagnostics.push_back(std::string(presumed.getFilename()) +
                                      ":" + std::to_string(presumed.getLine()) +
                                      ": error: " + msg);
        return;
      }
    }
    result.diagnostics.push_back("error: " + msg);
  }

  clang::ASTContext& ctx;
  SubsetResult& result;
};

} // namespace

SubsetResult enforceSubset(clang::ASTContext& context) {
  SubsetResult result;
  SubsetVisitor visitor(context, result);
  visitor.TraverseDecl(context.getTranslationUnitDecl());
  return result;
}

} // namespace arcanum
