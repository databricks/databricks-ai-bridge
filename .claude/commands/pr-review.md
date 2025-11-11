---
allowed-tools: Read, Skill, Bash, Grep, Glob, mcp__review__fetch_diff
argument-hint: [extra_context]
description: Review a GitHub pull request and display all issues found in Claude Code
---

# Review Pull Request

Automatically review a GitHub pull request and display all found issues in Claude Code with detailed analysis and suggestions.

## Usage

```
/pr-review [extra_context]
```

## Arguments

- `extra_context` (optional): Additional instructions or filtering context (e.g., focus on specific issues or areas)

## Examples

```
/pr-review                                    # Review all changes
/pr-review Please focus on security issues    # Focus on security
/pr-review Only review Python files           # Filter specific file types
/pr-review Check for performance issues       # Focus on specific concern
```

## Instructions

### 1. Auto-detect PR context

- First check for environment variables:
  - If `PR_NUMBER` and `GITHUB_REPOSITORY` are set, parse `GITHUB_REPOSITORY` as `owner/repo` and use `PR_NUMBER`
  - Then use `gh pr view <PR_NUMBER> --repo <owner/repo> --json 'title,body'` to retrieve the PR title and description
- Otherwise:
  - Use `gh pr view --json 'title,body,url,number'` to get PR info for the current branch
  - Parse the output to extract owner, repo, PR number, title, and description
- If neither method works, inform the user that no PR was found and exit

### 2. Fetch PR Diff

- Use `mcp__review__fetch_diff` tool to fetch the PR diff

### 3. Review Changed Lines

**Apply additional filtering** from user instructions if provided (e.g., focus on specific issues or areas)

Carefully examine **only the changed lines** (added or modified) in the diff for:

- Potential bugs and code quality issues
- Common mistakes

**Important**: Ignore unchanged/context lines and pre-existing code.

**Collect all issues** in a structured format before presenting them:

```
issues = [
  {
    "file": "path/to/file.py",
    "line": 42,
    "end_line": 45, // (optional, for multi-line issues)
    "severity": "error|warning|info", 
    "category": "bug|style|performance|security",
    "description": "Detailed issue description",
    "suggestion": "Recommended fix"
  }
]
```

### 4. Present Complete Analysis

After reviewing all changed files, display a comprehensive summary in Claude Code:

- **Group by file and severity** for clear organization
- **Display statistics**: Total issues, breakdown by severity and category
- **Use navigation-friendly format**: Include `file:line` references for easy IDE navigation
- **Show detailed descriptions** and suggested fixes for each issue

**Output format:**
```
## PR Review Results

**Total Issues Found: X**
- 🔴 Errors: X
- 🟡 Warnings: X  
- 🔵 Info: X

### path/to/file1.py
🔴 **Line 42-45 [Bug]:** Description here
   → Suggestion: Fix recommendation

🟡 **Line 67 [Style]:** Description here  
   → Suggestion: Fix recommendation

### path/to/file2.js
🔴 **Line 23 [Security]:** Description here
   → Suggestion: Fix recommendation
```

If **no issues found**, display: "✅ **No issues found** - All changes look good!"
