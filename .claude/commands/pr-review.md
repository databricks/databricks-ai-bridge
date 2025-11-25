---
allowed-tools: Read, Skill, Bash, Grep, Glob
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

### 2. Fetch PR Diff and Existing Comments

- Use `gh pr diff <PR_NUMBER> --repo <owner/repo>` to fetch the PR diff
- Use the `fetch_unresolved_comments` skill to get existing unresolved review comments:
  ```
  /skill fetch_unresolved_comments <owner> <repo> <PR_NUMBER>
  ```
- Parse the returned comment data to identify lines that already have unresolved issues to avoid duplicate reports

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
    "suggestion": "Recommended fix",
    "is_duplicate": false // true if similar issue already commented on this line
  }
]
```

**Important**: Before adding an issue to the list, check the existing unresolved comments from step 2. If there's already an unresolved comment on the same line(s) or covering a similar concern, mark the issue as duplicate to avoid redundant reports.

### 4. Present Complete Analysis

After reviewing all changed files, display a comprehensive summary in Claude Code:

- **Group by file and severity** for clear organization
- **Display statistics**: Total issues, breakdown by severity and category
- **Use navigation-friendly format**: Include `file:line` references for easy IDE navigation
- **Show detailed descriptions** and suggested fixes for each issue

**Output format:**
```
## PR Review Results

**Total Issues Found: X (Y new, Z already commented)**
- 🔴 Errors: X
- 🟡 Warnings: X  
- 🔵 Info: X

### path/to/file1.py:42-45
🔴 **[Bug]:** Description here
   → Suggestion: Fix recommendation

### path/to/file2.js:67
🟡 **[Style]:** Description here  
   → Suggestion: Fix recommendation

### path/to/file3.py:23
🔴 **[Security]:** Description here
   → Suggestion: Fix recommendation
   ⚠️ *Similar issue already commented on this PR*
```

**Format Notes:**
- Use `file:line` or `file:line-endline` format for direct IDE navigation
- Mark duplicate/already-commented issues with ⚠️ warning
- Group issues by file, with each issue as a separate section for easy clicking

If **no issues found**, display: "✅ **No issues found** - All changes look good!"
