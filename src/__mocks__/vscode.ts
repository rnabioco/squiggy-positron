/**
 * Mock VSCode API for testing
 */

export class TreeItem {
    label: string;
    collapsibleState: TreeItemCollapsibleState | number;
    tooltip?: string;
    contextValue?: string;
    iconPath?: ThemeIcon | string | { light: string; dark: string };
    command?: Command;

    constructor(label: string, collapsibleState: TreeItemCollapsibleState | number) {
        this.label = label;
        this.collapsibleState = collapsibleState;
    }
}

export enum TreeItemCollapsibleState {
    None = 0,
    Collapsed = 1,
    Expanded = 2,
}

export class EventEmitter<T> {
    private listeners: ((e: T) => void)[] = [];

    get event() {
        return (listener: (e: T) => void) => {
            this.listeners.push(listener);
            return { dispose: () => {} };
        };
    }

    fire(data: T) {
        this.listeners.forEach((l) => l(data));
    }
}

export class ThemeIcon {
    constructor(public id: string) {}
}

export interface Command {
    command: string;
    title: string;
    arguments?: unknown[];
}
