//! Fast filesystem operations for Harness Engineering CLI.
//!
//! Provides native-performance:
//! - Glob pattern matching with .gitignore awareness
//! - Grep/text search with regex support
//! - File metadata collection
//! - Content hashing
//! - Parallel traversal

use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use walkdir::WalkDir;
use ignore::WalkBuilder;
use globset::{Glob, GlobSetBuilder};
use rayon::prelude::*;

/// Fast glob search with .gitignore awareness.
///
/// Returns a list of matching file paths, respecting .gitignore rules.
/// Significantly faster than Python's pathlib.rglob for large directories.
#[pyfunction]
fn fast_glob(
    root: &str,
    pattern: &str,
    max_files: usize,
    respect_gitignore: bool,
    include_hidden: bool,
) -> PyResult<Vec<String>> {
    let root_path = Path::new(root);
    if !root_path.exists() {
        return Ok(Vec::new());
    }

    let mut builder = WalkBuilder::new(root_path);
    builder
        .max_depth(None)
        .follow_links(false)
        .standard_filters(respect_gitignore)
        .hidden(!include_hidden);

    // Set thread count for parallel traversal
    builder.threads(num_cpus::min(4));

    let glob = Glob::new(pattern)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
        .compile_matcher();

    let results: Vec<String> = builder
        .build()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .filter(|entry| {
            glob.is_match(entry.path())
        })
        .map(|entry| entry.path().to_string_lossy().to_string())
        .collect();

    if max_files > 0 && results.len() > max_files {
        Ok(results[..max_files].to_vec())
    } else {
        Ok(results)
    }
}

/// Fast recursive file listing with metadata.
///
/// Returns file metadata including path, size, mtime.
#[pyfunction]
fn fast_file_index(
    root: &str,
    max_files: usize,
    respect_gitignore: bool,
) -> PyResult<Vec<HashMap<String, PyObject>>> {
    let root_path = Path::new(root);
    if !root_path.exists() {
        return Ok(Vec::new());
    }

    let mut builder = WalkBuilder::new(root_path);
    builder
        .max_depth(None)
        .follow_links(false)
        .standard_filters(respect_gitignore)
        .hidden(false)
        .threads(num_cpus::min(4));

    let results: Vec<HashMap<String, PyObject>> = builder
        .build()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .take(if max_files > 0 { max_files } else { usize::MAX })
        .filter_map(|entry| {
            let metadata = entry.metadata().ok()?;
            let mut map = HashMap::new();
            map.insert("path".to_string(), entry.path().to_string_lossy().to_string().into());
            map.insert("size".to_string(), metadata.len().into());
            if let Ok(modified) = metadata.modified() {
                if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                    map.insert("mtime".to_string(), duration.as_secs_f64().into());
                }
            }
            map.insert("is_dir".to_string(), metadata.is_dir().into());
            Some(map)
        })
        .collect();

    Ok(results)
}

/// Fast grep/text search with regex support.
///
/// Searches files matching a pattern for lines containing the search term.
/// Returns list of {file, line, content} matches.
#[pyfunction]
fn fast_grep(
    root: &str,
    pattern: &str,
    path_filter: Option<&str>,
    max_results: usize,
    case_insensitive: bool,
    respect_gitignore: bool,
) -> PyResult<Vec<HashMap<String, String>>> {
    let root_path = Path::new(root);
    if !root_path.exists() {
        return Ok(Vec::new());
    }

    // Build regex pattern
    let case_flag = if case_insensitive { "(?i)" } else { "" };
    let full_pattern = format!("{}{}", case_flag, pattern);
    let regex = grep_regex::RegexMatcherBuilder::new()
        .case_insensitive(case_insensitive)
        .build(&pattern)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let mut builder = WalkBuilder::new(root_path);
    builder
        .max_depth(None)
        .follow_links(false)
        .standard_filters(respect_gitignore)
        .hidden(false)
        .threads(num_cpus::min(4));

    // Optional path filter
    let path_matcher = if let Some(filter) = path_filter {
        let glob = Glob::new(filter)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .compile_matcher();
        Some(glob)
    } else {
        None
    };

    let results: Arc<Mutex<Vec<HashMap<String, String>>>> = Arc::new(Mutex::new(Vec::new()));

    for entry in builder.build().filter_map(|e| e.ok()) {
        if results.lock().unwrap().len() >= max_results {
            break;
        }
        if !entry.file_type().is_file() {
            continue;
        }

        // Apply path filter
        if let Some(ref pm) = path_matcher {
            if !pm.is_match(entry.path()) {
                continue;
            }
        }

        let path = entry.path();

        // Skip binary files (quick heuristic)
        if let Ok(content) = std::fs::read(path) {
            if content.len() > 100_000_000 {
                continue; // Skip files > 100MB
            }
            // Check for null bytes (binary heuristic)
            if content.windows(1).any(|w| w[0] == 0) {
                continue;
            }

            if let Ok(text) = String::from_utf8(content) {
                for (line_num, line) in text.lines().enumerate() {
                    if let Some(mat) = regex.find_iter(line).next() {
                        let mut result = HashMap::new();
                        result.insert("file".to_string(), path.to_string_lossy().to_string());
                        result.insert("line".to_string(), (line_num + 1).to_string());
                        result.insert("content".to_string(), line.trim().to_string());
                        result.insert("match".to_string(), mat.as_str().to_string());

                        let mut results = results.lock().unwrap();
                        results.push(result);
                        if results.len() >= max_results {
                            break;
                        }
                    }
                }
            }
        }
    }

    let final_results = Arc::try_unwrap(results)
        .unwrap_or_else(|arc| arc.lock().unwrap().clone())
        .into_inner()
        .unwrap();

    Ok(final_results)
}

/// Compute content hash for file deduplication and change detection.
#[pyfunction]
fn fast_hash(path: &str) -> PyResult<String> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let content = std::fs::read(path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    Ok(format!("{:016x}", hasher.finish()))
}

/// Batch hash multiple files in parallel.
#[pyfunction]
fn fast_batch_hash(paths: Vec<String>) -> PyResult<HashMap<String, String>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let results: HashMap<String, String> = paths
        .par_iter()
        .filter_map(|path| {
            let content = std::fs::read(path).ok()?;
            let mut hasher = DefaultHasher::new();
            content.hash(&mut hasher);
            Some((path.clone(), format!("{:016x}", hasher.finish())))
        })
        .collect();

    Ok(results)
}

/// Count files matching a pattern (fast, metadata only).
#[pyfunction]
fn fast_count_files(
    root: &str,
    respect_gitignore: bool,
    extensions: Option<Vec<String>>,
) -> PyResult<usize> {
    let root_path = Path::new(root);
    if !root_path.exists() {
        return Ok(0);
    }

    let mut builder = WalkBuilder::new(root_path);
    builder
        .max_depth(None)
        .follow_links(false)
        .standard_filters(respect_gitignore)
        .hidden(false)
        .threads(num_cpus::min(4));

    let count = builder
        .build()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .filter(|entry| {
            if let Some(ref exts) = extensions {
                if let Some(ext) = entry.path().extension() {
                    exts.iter().any(|e| e == ext.to_string_lossy().as_ref())
                } else {
                    false
                }
            } else {
                true
            }
        })
        .count();

    Ok(count)
}

/// Python module definition.
#[pymodule]
fn harness_fs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_glob, m)?)?;
    m.add_function(wrap_pyfunction!(fast_file_index, m)?)?;
    m.add_function(wrap_pyfunction!(fast_grep, m)?)?;
    m.add_function(wrap_pyfunction!(fast_hash, m)?)?;
    m.add_function(wrap_pyfunction!(fast_batch_hash, m)?)?;
    m.add_function(wrap_pyfunction!(fast_count_files, m)?)?;
    Ok(())
}

/// Get number of available CPU cores.
fn num_cpus_min() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Helper module for internal use.
mod num_cpus {
    pub fn min(max: usize) -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get().min(max))
            .unwrap_or(1)
    }
}
