import { compare, valid } from "semver";

export interface LatestPiRelease {
	version: string;
	packageName?: string;
	note?: string;
}

export function comparePackageVersions(leftVersion: string, rightVersion: string): number | undefined {
	const left = valid(leftVersion.trim());
	const right = valid(rightVersion.trim());
	if (!left || !right) {
		return undefined;
	}
	return compare(left, right);
}

export function isNewerPackageVersion(candidateVersion: string, currentVersion: string): boolean {
	const comparison = comparePackageVersions(candidateVersion, currentVersion);
	if (comparison !== undefined) {
		return comparison > 0;
	}
	return candidateVersion.trim() !== currentVersion.trim();
}

export async function getLatestPiRelease(
	_currentVersion: string,
	_options: { timeoutMs?: number } = {},
): Promise<LatestPiRelease | undefined> {
	return undefined;
}

export async function getLatestPiVersion(
	_currentVersion: string,
	_options: { timeoutMs?: number } = {},
): Promise<string | undefined> {
	return undefined;
}

export async function checkForNewPiVersion(_currentVersion: string): Promise<LatestPiRelease | undefined> {
	return undefined;
}
