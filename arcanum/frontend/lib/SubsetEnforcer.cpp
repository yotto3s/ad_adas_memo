#include "arcanum/frontend/SubsetEnforcer.h"

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

/// Check whether a bit width is in the allowed set {8, 16, 32, 64}.
bool isAllowedWidth(uint64_t width) {
  switch (width) {
  case 8:
  case 16:
  case 32:
  case 64:
    return true;
  default:
    return false;
  }
}

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

  bool VisitForStmt(clang::ForStmt* stmt) {
    addDiagnostic(stmt->getBeginLoc(), "for loops are not allowed in Slice 1");
    return true;
  }

  bool VisitWhileStmt(clang::WhileStmt* stmt) {
    addDiagnostic(stmt->getBeginLoc(),
                  "while loops are not allowed in Slice 1");
    return true;
  }

  bool VisitDoStmt(clang::DoStmt* stmt) {
    addDiagnostic(stmt->getBeginLoc(),
                  "do-while loops are not allowed in Slice 1");
    return true;
  }

  bool VisitSwitchStmt(clang::SwitchStmt* stmt) {
    addDiagnostic(stmt->getBeginLoc(),
                  "switch statements are not allowed in Slice 1");
    return true;
  }

  bool VisitCXXForRangeStmt(clang::CXXForRangeStmt* stmt) {
    addDiagnostic(stmt->getBeginLoc(),
                  "range-based for loops are not allowed in Slice 1");
    return true;
  }

  bool VisitCallExpr(clang::CallExpr* expr) {
    // Skip calls in system headers (e.g., from <cstdint>)
    if (expr->getBeginLoc().isValid() &&
        ctx.getSourceManager().isInSystemHeader(expr->getBeginLoc())) {
      return true;
    }
    // Skip implicit/builtin calls
    if (const auto* callee = expr->getDirectCallee()) {
      if (callee->isImplicit() || callee->getBuiltinID() != 0) {
        return true;
      }
    }
    // Operator calls are not allowed
    if (llvm::isa<clang::CXXOperatorCallExpr>(expr)) {
      addDiagnostic(expr->getBeginLoc(),
                    "operator calls are not allowed in Slice 1");
      return true;
    }
    addDiagnostic(expr->getBeginLoc(),
                  "function calls are not allowed in Slice 1");
    return true;
  }

  // --- Cast validation (Slice 2) ---

  bool VisitCXXStaticCastExpr(clang::CXXStaticCastExpr* expr) {
    // static_cast is accepted; validate source and target are supported types
    checkType(expr->getTypeAsWritten(), expr->getBeginLoc());
    checkType(expr->getSubExpr()->getType(), expr->getBeginLoc());
    return true;
  }

  bool VisitCStyleCastExpr(clang::CStyleCastExpr* expr) {
    // Skip casts in system headers
    if (expr->getBeginLoc().isValid() &&
        ctx.getSourceManager().isInSystemHeader(expr->getBeginLoc())) {
      return true;
    }
    // Skip casts that originate from macro expansions (e.g., system macros
    // like INT32_C() that expand to C-style casts in user code)
    if (expr->getLParenLoc().isMacroID()) {
      return true;
    }
    addDiagnostic(expr->getBeginLoc(),
                  "use static_cast instead of C-style cast");
    return true;
  }

  bool VisitCXXReinterpretCastExpr(clang::CXXReinterpretCastExpr* expr) {
    addDiagnostic(expr->getBeginLoc(), "reinterpret_cast is not allowed");
    return true;
  }

  bool VisitCXXConstCastExpr(clang::CXXConstCastExpr* expr) {
    addDiagnostic(expr->getBeginLoc(), "const_cast is not allowed");
    return true;
  }

  bool VisitCXXDynamicCastExpr(clang::CXXDynamicCastExpr* expr) {
    addDiagnostic(expr->getBeginLoc(), "dynamic_cast is not allowed");
    return true;
  }

  // --- Mixed-type binary op validation (Slice 2) ---

  bool VisitBinaryOperator(clang::BinaryOperator* expr) {
    // Only check arithmetic and comparison operators on integer types.
    // Skip logical operators (&&, ||) which operate on bools, and
    // assignment operators.
    if (expr->isLogicalOp() || expr->isAssignmentOp()) {
      return true;
    }
    // Skip if in system headers
    if (expr->getBeginLoc().isValid() &&
        ctx.getSourceManager().isInSystemHeader(expr->getBeginLoc())) {
      return true;
    }

    // Get the actual source-level types by stripping implicit casts
    auto* lhs = expr->getLHS()->IgnoreParenImpCasts();
    auto* rhs = expr->getRHS()->IgnoreParenImpCasts();

    clang::QualType lhsType = lhs->getType().getCanonicalType();
    clang::QualType rhsType = rhs->getType().getCanonicalType();

    // Only check when both sides are integer types (not bool)
    if (!lhsType->isIntegerType() || lhsType->isBooleanType() ||
        !rhsType->isIntegerType() || rhsType->isBooleanType()) {
      return true;
    }

    // Skip the mixed-type check when either operand is an integer literal.
    // Integer literals have type 'int' (width 32) but are commonly used with
    // narrow types (int8_t, int16_t, uint8_t) after implicit integer
    // promotion.  Flagging `x + 1` or `x > 0` for narrow types would make
    // the tool unusable for real code (F5).
    if (llvm::isa<clang::IntegerLiteral>(lhs) ||
        llvm::isa<clang::IntegerLiteral>(rhs)) {
      return true;
    }

    uint64_t lhsWidth = ctx.getTypeSize(lhsType);
    uint64_t rhsWidth = ctx.getTypeSize(rhsType);
    bool lhsSigned = lhsType->isSignedIntegerType();
    bool rhsSigned = rhsType->isSignedIntegerType();

    if (lhsWidth != rhsWidth || lhsSigned != rhsSigned) {
      addDiagnostic(expr->getOperatorLoc(),
                    "operands must have matching types; use static_cast to "
                    "convert");
    }
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
    return std::any_of(
        stmt->child_begin(), stmt->child_end(),
        [this](const clang::Stmt* child) { return containsReturn(child); });
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
      for (const auto* it = compound->body_begin(); it != compound->body_end();
           ++it) {
        bool isLast = (std::next(it) == compound->body_end());
        // A bare return statement followed by more statements
        if (llvm::isa<clang::ReturnStmt>(*it) && !isLast) {
          return true;
        }
        // An if-without-else that contains a return, followed by more
        // statements -- this is the "guard clause" / early return pattern.
        if (!isLast) {
          if (const auto* ifStmt = llvm::dyn_cast<clang::IfStmt>(*it)) {
            if ((ifStmt->getElse() == nullptr) &&
                containsReturn(ifStmt->getThen())) {
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
    clang::QualType canonical = type.getCanonicalType();
    // Allow void
    if (canonical->isVoidType()) {
      return;
    }
    // Allow bool
    if (canonical->isBooleanType()) {
      return;
    }
    // Reject raw pointers
    if (canonical->isPointerType()) {
      addDiagnostic(loc, "raw pointer types are not allowed");
      return;
    }
    // Reject floating-point
    if (canonical->isFloatingType()) {
      addDiagnostic(loc, "floating-point types are not allowed in Slice 1");
      return;
    }
    // Accept any integer type with width in {8, 16, 32, 64}.
    // TODO(SC-1): Spec mandates rejecting platform-dependent types (int, short,
    // long) and directing users to <cstdint> fixed-width types.  The current
    // implementation intentionally accepts them if their width matches an
    // allowed width (e.g., int=32, short=16, long=64 on x86_64).  This is a
    // deliberate deviation for usability; revisit if spec conformance required.
    if (canonical->isIntegerType()) {
      uint64_t width = ctx.getTypeSize(canonical);
      if (isAllowedWidth(width)) {
        return;
      }
      addDiagnostic(loc, "integer type '" + type.getAsString() +
                             "' has unsupported width; use int8_t through "
                             "int64_t or uint8_t through uint64_t");
      return;
    }
    // Reject everything else
    addDiagnostic(loc, "type '" + type.getAsString() + "' is not allowed");
  }

  void addDiagnostic(clang::SourceLocation loc, const std::string& msg) {
    result.passed = false;
    auto& sm = ctx.getSourceManager();
    if (loc.isValid()) {
      auto presumed = sm.getPresumedLoc(loc);
      if (presumed.isValid()) {
        result.diagnostics.push_back(std::string(presumed.getFilename()) + ":" +
                                     std::to_string(presumed.getLine()) +
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
