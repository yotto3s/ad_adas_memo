#pragma once

#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceLocation.h"

#include <string>
#include <vector>

namespace arcanum {

/// Severity level for subset violations.
enum class ViolationSeverity {
    Error,   // Construct is completely forbidden
    Warning, // Construct is discouraged but not yet enforced
};

/// A single subset violation found during enforcement.
struct SubsetViolation {
    clang::SourceLocation location;
    ViolationSeverity severity;
    std::string description;
    std::string suggestion;
};

/// Result of running the subset enforcer on a translation unit.
struct SubsetEnforcerResult {
    std::vector<SubsetViolation> violations;

    bool passed() const { return violations.empty(); }
    std::size_t errorCount() const;
};

/// Enforces the Arcanum Safe C++ Subset on a Clang AST.
///
/// Walks the entire AST using RecursiveASTVisitor and rejects
/// constructs that are outside the allowed subset:
///   - Virtual functions and dynamic dispatch
///   - Raw pointers and pointer arithmetic
///   - Dynamic memory allocation (new/delete)
///   - Exceptions (throw/try/catch)
///   - goto and labels (non-annotation)
///   - reinterpret_cast, C-style casts, dynamic_cast
///   - Inline assembly
///
/// For Slice 1, the enforcer focuses on the minimal subset:
/// Types: int32_t, bool
/// Functions: non-template, non-recursive, single file, single return
/// Statements: variable decl, assignment, if/else, return
/// Expressions: arithmetic, comparison, logical operators
class SubsetEnforcer : public clang::RecursiveASTVisitor<SubsetEnforcer> {
public:
    explicit SubsetEnforcer(clang::ASTContext &context);

    /// Run the enforcer on the entire translation unit.
    SubsetEnforcerResult enforce();

    // --- RecursiveASTVisitor callbacks ---

    /// Reject virtual methods.
    bool VisitCXXMethodDecl(clang::CXXMethodDecl *decl);

    /// Reject throw expressions.
    bool VisitCXXThrowExpr(clang::CXXThrowExpr *expr);

    /// Reject try statements.
    bool VisitCXXTryStmt(clang::CXXTryStmt *stmt);

    /// Reject new expressions.
    bool VisitCXXNewExpr(clang::CXXNewExpr *expr);

    /// Reject delete expressions.
    bool VisitCXXDeleteExpr(clang::CXXDeleteExpr *expr);

    /// Reject goto statements.
    bool VisitGotoStmt(clang::GotoStmt *stmt);

    /// Reject reinterpret_cast.
    bool VisitCXXReinterpretCastExpr(clang::CXXReinterpretCastExpr *expr);

    /// Reject dynamic_cast.
    bool VisitCXXDynamicCastExpr(clang::CXXDynamicCastExpr *expr);

    /// Reject C-style casts.
    bool VisitCStyleCastExpr(clang::CStyleCastExpr *expr);

    /// Reject raw pointer declarations and pointer arithmetic.
    bool VisitVarDecl(clang::VarDecl *decl);

    /// Reject pointer parameters.
    bool VisitParmVarDecl(clang::ParmVarDecl *decl);

    /// Reject inline assembly.
    bool VisitGCCAsmStmt(clang::GCCAsmStmt *stmt);
    bool VisitMSAsmStmt(clang::MSAsmStmt *stmt);

    /// Reject setjmp/longjmp calls.
    bool VisitCallExpr(clang::CallExpr *expr);

private:
    clang::ASTContext &context_;
    std::vector<SubsetViolation> violations_;

    /// Add a violation with error severity.
    void addError(clang::SourceLocation loc, std::string description,
                  std::string suggestion);

    /// Add a violation with warning severity.
    void addWarning(clang::SourceLocation loc, std::string description,
                    std::string suggestion);

    /// Check if a source location is in the main file (skip system headers).
    bool isInMainFile(clang::SourceLocation loc) const;

    /// Check if a type is a raw pointer type.
    bool isRawPointerType(clang::QualType type) const;
};

} // namespace arcanum
