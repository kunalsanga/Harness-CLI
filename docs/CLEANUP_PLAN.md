## Files/Folders Proposed for Removal

| Path | Reason | Evidence | Risk |
|------|--------|----------|------|
| `.claude/` | Temporary IDE artifact | Untracked directory, clearly an IDE/agent artifact | SAFE |
| `all_files_list.txt` | Generated dump | Untracked plain text dump | SAFE |
| `architecture_dump.txt` | Generated dump | Untracked plain text dump | SAFE |
| `filtered_files_list.txt` | Generated dump | Untracked plain text dump | SAFE |
| `get_tree.py` | Temporary scratch script | Untracked scratch script | SAFE |
| `git_history.txt` | Generated dump | Untracked plain text dump | SAFE |
| `harness-full-project.txt` | Generated dump | Untracked plain text dump | SAFE |
| `project-tree.txt` | Generated dump | Untracked plain text dump | SAFE |
| `pytest_full_output.txt` | Generated dump | Untracked plain text dump | SAFE |
| `rewrite_audit.py` | Temporary scratch script | Untracked scratch script | SAFE |
| `scratch_analyzer.py` | Temporary scratch script | Untracked scratch script | SAFE |
| `scratch_audit.py` | Temporary scratch script | Untracked scratch script | SAFE |
| `scratch_audit2.py` | Temporary scratch script | Untracked scratch script | SAFE |
| `scratch_audit3.py` | Temporary scratch script | Untracked scratch script | SAFE |
| `scratch_audit4.py` | Temporary scratch script | Untracked scratch script | SAFE |
| `scratch_audit5.py` | Temporary scratch script | Untracked scratch script | SAFE |
| `scratch_box.py` | Temporary scratch script | Untracked scratch script | SAFE |
| `scratch_conout.py` | Temporary scratch script | Untracked scratch script | SAFE |
| `scratch_ctypes.py` | Temporary scratch script | Untracked scratch script | SAFE |
| `scratch_prompt.py` | Temporary scratch script | Untracked scratch script | SAFE |
| `scratch_pt.py` | Temporary scratch script | Untracked scratch script | SAFE |
| `scratch_size.py` | Temporary scratch script | Untracked scratch script | SAFE |
| `src_tree.txt` | Generated dump | Untracked plain text dump | SAFE |
| `test_list.txt` | Generated dump | Untracked plain text dump | SAFE |
| `tree_output.txt` | Generated dump | Untracked plain text dump | SAFE |
| `utf8_files.txt` | Generated dump | Untracked plain text dump | SAFE |
| `__pycache__/*` | Build/Python cache | Standard Python cache directories | SAFE |
| `.pytest_cache/*` | Test cache | Standard pytest cache | SAFE |
| `.ruff_cache/*` | Linter cache | Standard ruff cache | SAFE |
| `.mypy_cache/*` | Type checker cache | Standard mypy cache | SAFE |
| `docs/CURRENT_STATE_FORENSIC_AUDIT.md` | Audit document | Meaningful audit documentation | NEEDS REVIEW |
| `docs/HARNESS_2_0_ARCHITECTURE.md` | Architecture document | Important architecture documentation | LOW RISK |
| `src/harness_core/cli/conversation.py` | Source file | Actively imported by CLI files | LOW RISK |
| `tests/unit/test_canonical_runtime.py` | Test file | Contains tests for canonical runtime | LOW RISK |
| `tests/unit/test_forensic_audit_fixes.py` | Test file | Contains tests for forensic audit | LOW RISK |
| `tests/unit/test_ux_interaction.py` | Test file | Contains tests for UX interactions | LOW RISK |
