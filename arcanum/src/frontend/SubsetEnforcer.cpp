#include "frontend/SubsetEnforcer.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceManager.h"

#include <string>

namespace arcanum {
namespace {

class SubsetVisitor : public clang::RecursiveASTVisitor<SubsetVisitor> {
public:
  explicit SubsetVisitor(clang::ASTContext& ctx, SubsetResult& result)
      : ctx_(ctx), result_(result) {}

  bool VisitFunctionDecl(clang::FunctionDecl* decl) {
    if (!decl->hasBody()) {
      return true; // Skip declarations without bodies
    }
    // Skip compiler-generated functions
    if (decl->isImplicit()) {
      return true;
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
      auto width = ctx_.getTypeSize(type);
      if (width == 32 && type->isSignedIntegerType()) {
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
    result_.passed = false;
    auto& sm = ctx_.getSourceManager();
    if (loc.isValid()) {
      auto presumed = sm.getPresumedLoc(loc);
      if (presumed.isValid()) {
        result_.diagnostics.push_back(
            std::string(presumed.getFilename()) + ":" +
            std::to_string(presumed.getLine()) + ": error: " + msg);
        return;
      }
    }
    result_.diagnostics.push_back("error: " + msg);
  }

  clang::ASTContext& ctx_;
  SubsetResult& result_;
};

} // namespace

SubsetResult enforceSubset(clang::ASTContext& context) {
  SubsetResult result;
  SubsetVisitor visitor(context, result);
  visitor.TraverseDecl(context.getTranslationUnitDecl());
  return result;
}

} // namespace arcanum
