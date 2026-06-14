#!/bin/bash

# Function to get current branch or fallback
get_branch() {
    git symbolic-ref --short HEAD 2>/dev/null || echo "main"
}

git add .
if ! git diff-index --quiet HEAD; then
    MAIN_BRANCH=$(get_branch)
    git commit -m "chore: sync submodules and main repo"
    
    if git pull --rebase origin "$MAIN_BRANCH"; then
        git push origin "$MAIN_BRANCH"
    else
        git rebase --abort 2>/dev/null
        echo "Failed to sync main repo. Manual intervention may be required."
    fi
else
    echo "No changes in main repo"
fi

echo "All synced."
