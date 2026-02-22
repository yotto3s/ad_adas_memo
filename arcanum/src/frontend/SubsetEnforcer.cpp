#include "SubsetEnforcer.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceManager.h"

#include <algorithm>

namespace arcanum {

// --- SubsetEnforcerResult ---

std::size_t SubsetEnforcerResult::errorCount() const {
    return static_cast<std::size_t>(std::count_if(
        violations.begin(), violations.end(), [](const SubsetViolation &v) {
            return v.severity == ViolationSeverity::Error;
        }));
}

// --- SubsetEnforcer ---

SubsetEnforcer::SubsetEnforcer(clang::ASTContext &context)
    : context_(context) {}

SubsetEnforcerResult SubsetEnforcer::enforce() {
    violations_.clear();
    TraverseDecl(context_.getTranslationUnitDecl());
    return SubsetEnforcerResult{std::move(violations_)};
}

bool SubsetEnforcer::VisitCXXMethodDecl(clang::CXXMethodDecl *decl) {
    if (!isInMainFile(decl->getLocation()))
        return true;

    if (decl->isVirtual()) {
        addError(decl->getLocation(),
                 "virtual function not allowed in safe C++ subset",
                 "use template or std::variant for dispatch");
    }
    return true;
}

bool SubsetEnforcer::VisitCXXThrowExpr(clang::CXXThrowExpr *expr) {
    if (!isInMainFile(expr->getThrowLoc()))
        return true;

    addError(expr->getThrowLoc(),
             "throw expression not allowed in safe C++ subset",
             "use std::expected<T, E> for error handling");
    return true;
}

bool SubsetEnforcer::VisitCXXTryStmt(clang::CXXTryStmt *stmt) {
    if (!isInMainFile(stmt->getTryLoc()))
        return true;

    addError(stmt->getTryLoc(),
             "try/catch not allowed in safe C++ subset",
             "use std::expected<T, E> for error handling");
    return true;
}

bool SubsetEnforcer::VisitCXXNewExpr(clang::CXXNewExpr *expr) {
    if (!isInMainFile(expr->getBeginLoc()))
        return true;

    addError(expr->getBeginLoc(),
             "new expression not allowed in safe C++ subset",
             "use stack allocation, std::array, or arena allocators");
    return true;
}

bool SubsetEnforcer::VisitCXXDeleteExpr(clang::CXXDeleteExpr *expr) {
    if (!isInMainFile(expr->getBeginLoc()))
        return true;

    addError(expr->getBeginLoc(),
             "delete expression not allowed in safe C++ subset",
             "use stack allocation, std::array, or arena allocators");
    return true;
}

bool SubsetEnforcer::VisitGotoStmt(clang::GotoStmt *stmt) {
    if (!isInMainFile(stmt->getGotoLoc()))
        return true;

    addError(stmt->getGotoLoc(),
             "goto not allowed in safe C++ subset",
             "use structured control flow (if, for, while, break, continue)");
    return true;
}

bool SubsetEnforcer::VisitCXXReinterpretCastExpr(
    clang::CXXReinterpretCastExpr *expr) {
    if (!isInMainFile(expr->getBeginLoc()))
        return true;

    addError(expr->getBeginLoc(),
             "reinterpret_cast not allowed in safe C++ subset",
             "use std::bit_cast<T> for type-safe reinterpretation");
    return true;
}

bool SubsetEnforcer::VisitCXXDynamicCastExpr(
    clang::CXXDynamicCastExpr *expr) {
    if (!isInMainFile(expr->getBeginLoc()))
        return true;

    addError(expr->getBeginLoc(),
             "dynamic_cast not allowed in safe C++ subset",
             "use std::variant with std::visit for type-safe dispatch");
    return true;
}

bool SubsetEnforcer::VisitCStyleCastExpr(clang::CStyleCastExpr *expr) {
    if (!isInMainFile(expr->getBeginLoc()))
        return true;

    // Allow implicit casts that Clang wraps in CStyleCastExpr internally.
    // Only reject explicit C-style casts written by the user.
    if (expr->getCastKind() == clang::CK_NoOp)
        return true;

    addError(expr->getLParenLoc(),
             "C-style cast not allowed in safe C++ subset",
             "use static_cast<T> for explicit conversions");
    return true;
}

bool SubsetEnforcer::VisitVarDecl(clang::VarDecl *decl) {
    if (!isInMainFile(decl->getLocation()))
        return true;

    // Skip ParmVarDecl -- handled separately.
    if (llvm::isa<clang::ParmVarDecl>(decl))
        return true;

    if (isRawPointerType(decl->getType())) {
        addError(decl->getLocation(),
                 "raw pointer variable not allowed in safe C++ subset",
                 "use references, std::span, or std::array");
    }
    return true;
}

bool SubsetEnforcer::VisitParmVarDecl(clang::ParmVarDecl *decl) {
    if (!isInMainFile(decl->getLocation()))
        return true;

    if (isRawPointerType(decl->getType())) {
        addError(decl->getLocation(),
                 "raw pointer parameter not allowed in safe C++ subset",
                 "use references, std::span<T>, or std::array reference");
    }
    return true;
}

bool SubsetEnforcer::VisitGCCAsmStmt(clang::GCCAsmStmt *stmt) {
    if (!isInMainFile(stmt->getAsmLoc()))
        return true;

    addError(stmt->getAsmLoc(),
             "inline assembly not allowed in safe C++ subset",
             "use //@ trusted functions for hardware access");
    return true;
}

bool SubsetEnforcer::VisitMSAsmStmt(clang::MSAsmStmt *stmt) {
    if (!isInMainFile(stmt->getAsmLoc()))
        return true;

    addError(stmt->getAsmLoc(),
             "inline assembly not allowed in safe C++ subset",
             "use //@ trusted functions for hardware access");
    return true;
}

bool SubsetEnforcer::VisitCallExpr(clang::CallExpr *expr) {
    if (!isInMainFile(expr->getBeginLoc()))
        return true;

    if (auto *callee = expr->getDirectCallee()) {
        llvm::StringRef name = callee->getName();
        if (name == "setjmp" || name == "longjmp" || name == "_setjmp" ||
            name == "_longjmp") {
            addError(expr->getBeginLoc(),
                     "setjmp/longjmp not allowed in safe C++ subset",
                     "use structured control flow");
        }
    }
    return true;
}

// --- Private helpers ---

void SubsetEnforcer::addError(clang::SourceLocation loc,
                              std::string description,
                              std::string suggestion) {
    violations_.push_back(SubsetViolation{
        loc, ViolationSeverity::Error, std::move(description),
        std::move(suggestion)});
}

void SubsetEnforcer::addWarning(clang::SourceLocation loc,
                                std::string description,
                                std::string suggestion) {
    violations_.push_back(SubsetViolation{
        loc, ViolationSeverity::Warning, std::move(description),
        std::move(suggestion)});
}

bool SubsetEnforcer::isInMainFile(clang::SourceLocation loc) const {
    if (loc.isInvalid())
        return false;
    return context_.getSourceManager().isInMainFile(loc);
}

bool SubsetEnforcer::isRawPointerType(clang::QualType type) const {
    return type->isPointerType() && !type->isFunctionPointerType();
}

} // namespace arcanum
