/**
 * Mock VSCode API for testing
 */

export class TreeItem {
    label: string;
    collapsibleState: any;
    tooltip?: string;
    contextValue?: string;
    iconPath?: any;
    command?: any;

    constructor(label: string, collapsibleState: any) {
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
    private listeners: ((e: T) => any)[] = [];

    get event() {
        return (listener: (e: T) => any) => {
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
    arguments?: any[];
}
