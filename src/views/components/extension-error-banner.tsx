/**
 * ExtensionErrorBanner
 *
 * Listens for `{ type: 'error', error: {...} }` messages posted by the
 * extension host (via BaseWebviewProvider.sendErrorToWebview) and renders a
 * dismissible banner above the panel content.
 *
 * Wrapping every panel with this component ensures that failures surfaced by
 * the extension (kernel errors, failed updates, command handler errors) are
 * actually shown to the user instead of being silently dropped.
 */

import * as React from 'react';

interface ExtensionError {
    message: string;
    context?: string;
    type?: string;
}

const bannerStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: '8px',
    padding: '6px 10px',
    margin: '4px',
    borderRadius: '3px',
    fontSize: '12px',
    lineHeight: 1.4,
    background: 'var(--vscode-inputValidation-errorBackground, rgba(190, 17, 0, 0.2))',
    border: '1px solid var(--vscode-inputValidation-errorBorder, #be1100)',
    color: 'var(--vscode-foreground)',
};

const dismissStyle: React.CSSProperties = {
    flex: '0 0 auto',
    background: 'transparent',
    border: 'none',
    color: 'var(--vscode-foreground)',
    cursor: 'pointer',
    fontSize: '14px',
    lineHeight: 1,
    padding: '0 2px',
};

export function ExtensionErrorBanner({
    children,
}: {
    children: React.ReactNode;
}): React.ReactElement {
    const [error, setError] = React.useState<ExtensionError | null>(null);

    React.useEffect(() => {
        const handler = (event: MessageEvent) => {
            const message = event.data;
            if (message && message.type === 'error' && message.error) {
                setError(message.error as ExtensionError);
            }
        };
        window.addEventListener('message', handler);
        return () => window.removeEventListener('message', handler);
    }, []);

    return (
        <>
            {error && (
                <div role="alert" style={bannerStyle}>
                    <span>
                        {error.context ? `${error.context}: ` : ''}
                        {error.message}
                    </span>
                    <button
                        type="button"
                        aria-label="Dismiss error"
                        title="Dismiss"
                        style={dismissStyle}
                        onClick={() => setError(null)}
                    >
                        ✕
                    </button>
                </div>
            )}
            {children}
        </>
    );
}
