---
allowed-tools: Read, Skill, Bash, Grep, Glob, mcp__review__fetch_diff, mcp__review__add_pr_review_comment
argument-hint: [extra_context]
description: Review a GitHub pull request and add review comments for issues found
---

# Review Pull Request

Automatically review a GitHub pull request and provide feedback on code quality, style guide violations, and potential bugs.

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

### 4. Decision Point

- If **no issues found** → Output "No issues found" and exit successfully
- If **issues found** → Output all of them and exit successfully

**Tool parameters:**

- Single-line comment: Set `subject_type` to `line`, specify `line`
- Multi-line comment: Set `subject_type` to `line`, specify both `start_line` and `line`
