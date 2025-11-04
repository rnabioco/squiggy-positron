/**
 * SessionCore Component
 *
 * Top-level React component for the session manager panel.
 * Manages session state and handles communication with the extension host.
 */

import * as React from 'react';
import { vscode } from './vscode-api';
import './squiggy-session-core.css';

interface SessionState {
    hasSamples: boolean;
    hasSavedSession: boolean;
    sampleCount: number;
    sampleNames: string[];
}

export const SessionCore: React.FC = () => {
    const [state, setState] = React.useState<SessionState>({
        hasSamples: false,
        hasSavedSession: false,
        sampleCount: 0,
        sampleNames: [],
    });

    // Handle messages from extension
    React.useEffect(() => {
        const messageHandler = (event: MessageEvent) => {
            const message = event.data;

            if (message.type === 'updateSession') {
                setState({
                    hasSamples: message.hasSamples,
                    hasSavedSession: message.hasSavedSession,
                    sampleCount: message.sampleCount,
                    sampleNames: message.sampleNames,
                });
            }
        };

        window.addEventListener('message', messageHandler);

        // Notify extension that webview is ready
        vscode.postMessage({ type: 'ready' });

        return () => window.removeEventListener('message', messageHandler);
    }, []);

    const handleLoadDemo = () => {
        vscode.postMessage({ type: 'loadDemo' });
    };

    const handleSave = () => {
        vscode.postMessage({ type: 'save' });
    };

    const handleRestore = () => {
        vscode.postMessage({ type: 'restore' });
    };

    const handleExport = () => {
        vscode.postMessage({ type: 'export' });
    };

    const handleImport = () => {
        vscode.postMessage({ type: 'import' });
    };

    const handleClear = () => {
        vscode.postMessage({ type: 'clear' });
    };

    return (
        <div className="session-container">
            {/* Demo Session Card */}
            <div className="demo-card">
                <h3 className="demo-title">
                    <span className="icon">🚀</span> Try Squiggy with Demo Data
                </h3>
                <p className="demo-description">
                    Explore 180 yeast tRNA reads with base annotations
                </p>
                <button className="btn btn-primary btn-full" onClick={handleLoadDemo}>
                    <span className="icon">▶</span> Load Demo Session
                </button>
                <div className="demo-info">
                    <span className="icon">ℹ️</span> Uses packaged test data - no files needed!
                </div>
            </div>

            {/* Current Session Status */}
            <div className="section">
                <h4 className="section-title">Current Session</h4>
                <div className="status-box">
                    {state.hasSamples ? (
                        <>
                            <span className="icon">✓</span> {state.sampleCount} sample(s) loaded:{' '}
                            {state.sampleNames.join(', ')}
                        </>
                    ) : (
                        <>
                            <span className="icon">○</span> No data loaded
                        </>
                    )}
                </div>
            </div>

            {/* Session Actions */}
            <div className="section">
                <h4 className="section-title">Session Actions</h4>
                <button
                    className="btn btn-full"
                    onClick={handleSave}
                    disabled={!state.hasSamples}
                >
                    <span className="icon">💾</span> Save Session
                </button>
                <button
                    className="btn btn-secondary btn-full"
                    onClick={handleRestore}
                    disabled={!state.hasSavedSession}
                >
                    <span className="icon">📜</span> Restore Session
                </button>
            </div>

            {/* Import/Export */}
            <div className="section">
                <h4 className="section-title">Import/Export</h4>
                <button
                    className="btn btn-secondary btn-full"
                    onClick={handleExport}
                    disabled={!state.hasSamples}
                >
                    <span className="icon">📤</span> Export to File
                </button>
                <button className="btn btn-secondary btn-full" onClick={handleImport}>
                    <span className="icon">📥</span> Import from File
                </button>
            </div>

            {/* Cleanup */}
            <div className="section">
                <button
                    className="btn btn-secondary btn-full"
                    onClick={handleClear}
                    disabled={!state.hasSavedSession}
                >
                    <span className="icon">🗑️</span> Clear Saved Session
                </button>
            </div>
        </div>
    );
};
