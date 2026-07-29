#!/usr/bin/env node

/**
 * Package-manager CLI (air-gapped)
 * Supports: install, remove, uninstall, list
 * Removed: config-selector, version-check, windows-self-update
 */

import { parseArgs } from "./cli/args.ts";
import { APP_NAME } from "./config.ts";
import { SettingsManager } from "./core/settings-manager.ts";

async function main(): Promise<void> {
	const args = parseArgs(process.argv.slice(2));
	const settings = new SettingsManager();

	if (args.help || args.messages.length === 0) {
		console.log(`${APP_NAME} package manager

Commands:
  install <source> [-l]    Install extension source and add to settings
  remove <source> [-l]     Remove extension source from settings
  uninstall <source> [-l]  Alias for remove
  list                     List installed extensions from settings

Options:
  -l, --local              Operate on project-local settings
  -h, --help               Show this help
`);
		process.exit(0);
	}

	const command = args.messages[0];
	const target = args.messages[1];
	const local = args.unknownFlags.has("local") || args.unknownFlags.has("l");

	switch (command) {
		case "install": {
			if (!target) {
				console.error("Error: install requires a source argument");
				process.exit(1);
			}
			await settings.installExtension(target, { local });
			console.log(`Installed ${target}`);
			break;
		}
		case "remove":
		case "uninstall": {
			if (!target) {
				console.error("Error: remove requires a source argument");
				process.exit(1);
			}
			await settings.removeExtension(target, { local });
			console.log(`Removed ${target}`);
			break;
		}
		case "list": {
			const extensions = await settings.listExtensions({ local });
			for (const ext of extensions) {
				console.log(ext);
			}
			break;
		}
		default: {
			console.error(`Unknown command: ${command}`);
			process.exit(1);
		}
	}
}

main().catch((err) => {
	console.error(err);
	process.exit(1);
});
