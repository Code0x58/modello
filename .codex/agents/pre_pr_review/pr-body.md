# PR title and body review

The audience is a reviewer assessing the final state of the branch against its
target. Derive the description from the reviewed net diff and current behavior.
Commit history and the implementation conversation are not the product being
reviewed.

- Lead with the problem or capability and the resulting behavior. Make feature
  additions as clear and concise as bug fixes: what can a caller do now, and
  under what conditions?
- Include a minimal reproducing example when it clarifies the change. For a
  feature, show the smallest useful call and outcome; for a fix, show the
  triggering input and corrected result, with the previous behavior only where
  it helps explain the bug. Reuse established models and names where practical.
- Explain implementation details only when they help assess correctness, scope,
  compatibility, or a material tradeoff. Mention limitations that affect use.
- Summarize the validation actually performed and any meaningful gaps. Match
  each claim to the reviewed branch state; do not present planned checks as
  passing or imply coverage broader than the evidence supports.
- If the branch records an upstream failure, explain the affected capability
  and link the verified upstream issue or fix when available. Distinguish that
  limitation from a Modello defect, and verify any claim that it is resolved.
- On PR updates, rewrite the title and body around the final implementation.
  Remove abandoned approaches, superseded APIs, intermediate fixes, and a diary
  of commits or review rounds unless they explain a surviving tradeoff.
- Scale detail to the change. A small fix can need only a short explanation and
  validation; larger changes can use a few clear sections or examples. Follow
  a repository PR template if one exists.

Return a suggested title and a complete replacement body ready for the parent
agent to use. If no current body or draft was supplied, say the suggestion was
constructed from the branch. Flag unsupported claims separately. Do not create
or update the remote PR from the review subagent.
