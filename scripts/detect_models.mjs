import fs from "node:fs";
import path from "node:path";
import os from "node:os";

const homeDir = os.homedir();
const rootDir = process.cwd();

function hasConfigFile(dirPath) {
	if (!dirPath || !fs.existsSync(dirPath)) return false;
	return fs.existsSync(path.join(dirPath, "config.json"));
}

function findHighestVersionSubdir(basePath) {
	if (!fs.existsSync(basePath)) return undefined;
	try {
		const items = fs.readdirSync(basePath);
		const numericDirs = items
			.filter((item) => {
				const full = path.join(basePath, item);
				return fs.statSync(full).isDirectory();
			})
			.sort((a, b) => {
				const numA = Number.parseInt(a, 10);
				const numB = Number.parseInt(b, 10);
				if (!Number.isNaN(numA) && !Number.isNaN(numB)) return numB - numA;
				return b.localeCompare(a);
			});
		for (const dir of numericDirs) {
			const candidate = path.join(basePath, dir);
			if (hasConfigFile(candidate)) return candidate;
		}
		if (hasConfigFile(basePath)) return basePath;
	} catch {
		// Ignore read errors
	}
	return undefined;
}

function isSupportedMlxConfig(dirPath) {
	if (!dirPath || !fs.existsSync(dirPath)) return false;
	const configPath = path.join(dirPath, "config.json");
	return fs.existsSync(configPath);
}

function detectMlxModelPath() {
	if (process.env.OPENCODE_MLX_MODEL_PATH && isSupportedMlxConfig(process.env.OPENCODE_MLX_MODEL_PATH)) {
		return process.env.OPENCODE_MLX_MODEL_PATH;
	}
	const localPath = path.join(rootDir, "models", "mlx_model");
	if (isSupportedMlxConfig(localPath)) return localPath;

	// Check KaggleHub Qwen3 4-bit MLX downloads first
	const kaggleQwenBase = path.join(
		homeDir, ".cache", "kagglehub", "models",
		"coolgamerz", "qwen3-6-27b-4bitmlx", "other", "default"
	);
	if (fs.existsSync(kaggleQwenBase)) {
		const found = findHighestVersionSubdir(kaggleQwenBase);
		if (found && isSupportedMlxConfig(found)) return found;
	}

	return localPath;
}

function detectBgeEmbedPath() {
	if (process.env.OPENCODE_BGE_EMBED_PATH && hasConfigFile(process.env.OPENCODE_BGE_EMBED_PATH)) {
		return process.env.OPENCODE_BGE_EMBED_PATH;
	}
	const kaggleBgePath = path.join(homeDir, ".cache", "kagglehub", "models", "jonathanchan", "baai", "transformers", "bge-large-en-v1.5");
	const foundKaggle = findHighestVersionSubdir(kaggleBgePath);
	if (foundKaggle) return foundKaggle;

	const localPath = path.join(rootDir, "models", "bge-large-en-v1.5");
	if (hasConfigFile(localPath)) return localPath;

	return path.join(rootDir, "models", "bge-large-en-v1.5");
}

function detectVectorsFile() {
	if (process.env.OPENCODE_VECTORS_FILE && fs.existsSync(process.env.OPENCODE_VECTORS_FILE)) {
		return process.env.OPENCODE_VECTORS_FILE;
	}
	const dataPath = path.join(rootDir, "data", "code_vectors.npy");
	if (fs.existsSync(dataPath)) return dataPath;
	const rootPath = path.join(rootDir, "code_vectors.npy");
	if (fs.existsSync(rootPath)) return rootPath;
	return dataPath;
}

function detectMetadataFile() {
	if (process.env.OPENCODE_METADATA_FILE && fs.existsSync(process.env.OPENCODE_METADATA_FILE)) {
		return process.env.OPENCODE_METADATA_FILE;
	}
	const dataPath = path.join(rootDir, "data", "code_metadata.json");
	if (fs.existsSync(dataPath)) return dataPath;
	const rootPath = path.join(rootDir, "code_metadata.json");
	if (fs.existsSync(rootPath)) return rootPath;
	return dataPath;
}

const mlxModelPath = detectMlxModelPath();
const bgeEmbedPath = detectBgeEmbedPath();
const vectorsFile = detectVectorsFile();
const metadataFile = detectMetadataFile();

const envContent = [
	`export OPENCODE_MLX_MODEL_PATH="${mlxModelPath}"`,
	`export OPENCODE_BGE_EMBED_PATH="${bgeEmbedPath}"`,
	`export OPENCODE_VECTORS_FILE="${vectorsFile}"`,
	`export OPENCODE_METADATA_FILE="${metadataFile}"`,
].join("\n") + "\n";

fs.writeFileSync(path.join(rootDir, ".env.opencode"), envContent);
fs.writeFileSync(path.join(rootDir, "env.sh"), envContent);

console.log("\n========================================================");
console.log("  OpenCode Auto-Detected Model & Vector Index Paths    ");
console.log("========================================================");
console.log(envContent);
console.log("Wrote environment configuration to .env.opencode and env.sh");
console.log("Run 'source env.sh' or source .env.opencode to export these variables in your active shell.\n");
