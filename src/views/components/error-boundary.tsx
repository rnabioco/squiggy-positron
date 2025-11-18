/*---------------------------------------------------------------------------------------------
 *  Copyright (C) 2024 Posit Software, PBC. All rights reserved.
 *  Licensed under the Elastic License 2.0. See LICENSE.txt for license information.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react';

interface ErrorBoundaryProps {
    children: React.ReactNode;
    fallback?: (error: Error, errorInfo: React.ErrorInfo) => React.ReactNode;
}

interface ErrorBoundaryState {
    hasError: boolean;
    error?: Error;
    errorInfo?: React.ErrorInfo;
}

/**
 * ErrorBoundary component that catches React rendering errors and displays a fallback UI.
 *
 * Prevents entire panel from crashing when a component error occurs.
 * Follows React error boundary pattern: https://react.dev/reference/react/Component#catching-rendering-errors-with-an-error-boundary
 *
 * Usage:
 * ```tsx
 * <ErrorBoundary>
 *   <MyComponent />
 * </ErrorBoundary>
 * ```
 *
 * With custom fallback:
 * ```tsx
 * <ErrorBoundary fallback={(error) => <div>Custom error: {error.message}</div>}>
 *   <MyComponent />
 * </ErrorBoundary>
 * ```
 */
export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
    constructor(props: ErrorBoundaryProps) {
        super(props);
        this.state = { hasError: false };
    }

    static getDerivedStateFromError(error: Error): ErrorBoundaryState {
        // Update state so the next render will show the fallback UI
        return {
            hasError: true,
            error,
        };
    }

    componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
        // Log error details for debugging
        console.error('ErrorBoundary caught an error:', error);
        console.error('Component stack:', errorInfo.componentStack);

        // Update state with error info
        this.setState({
            errorInfo,
        });
    }

    private handleReset = (): void => {
        this.setState({
            hasError: false,
            error: undefined,
            errorInfo: undefined,
        });
    };

    render(): React.ReactNode {
        if (this.state.hasError && this.state.error) {
            // Custom fallback provided
            if (this.props.fallback) {
                return this.props.fallback(this.state.error, this.state.errorInfo!);
            }

            // Default fallback UI
            return (
                <div
                    style={{
                        padding: '20px',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        height: '100%',
                        fontFamily: 'var(--vscode-font-family)',
                        color: 'var(--vscode-foreground)',
                    }}
                >
                    <div
                        style={{
                            maxWidth: '600px',
                            textAlign: 'center',
                        }}
                    >
                        <h3
                            style={{
                                color: 'var(--vscode-errorForeground)',
                                marginBottom: '16px',
                            }}
                        >
                            Something went wrong
                        </h3>
                        <p
                            style={{
                                marginBottom: '16px',
                                opacity: 0.8,
                            }}
                        >
                            An error occurred while rendering this panel. This might be due to
                            invalid data or a component issue.
                        </p>
                        <details
                            style={{
                                marginBottom: '16px',
                                textAlign: 'left',
                                padding: '12px',
                                backgroundColor: 'var(--vscode-editor-background)',
                                borderRadius: '4px',
                                border: '1px solid var(--vscode-panel-border)',
                            }}
                        >
                            <summary
                                style={{
                                    cursor: 'pointer',
                                    marginBottom: '8px',
                                    fontWeight: 'bold',
                                }}
                            >
                                Error Details
                            </summary>
                            <pre
                                style={{
                                    margin: 0,
                                    fontSize: '12px',
                                    overflow: 'auto',
                                    maxHeight: '200px',
                                }}
                            >
                                {this.state.error.message}
                                {'\n\n'}
                                {this.state.error.stack}
                            </pre>
                        </details>
                        <button
                            onClick={this.handleReset}
                            style={{
                                padding: '8px 16px',
                                backgroundColor: 'var(--vscode-button-background)',
                                color: 'var(--vscode-button-foreground)',
                                border: 'none',
                                borderRadius: '2px',
                                cursor: 'pointer',
                                fontSize: '13px',
                            }}
                            onMouseOver={(e) => {
                                e.currentTarget.style.backgroundColor =
                                    'var(--vscode-button-hoverBackground)';
                            }}
                            onMouseOut={(e) => {
                                e.currentTarget.style.backgroundColor =
                                    'var(--vscode-button-background)';
                            }}
                        >
                            Try Again
                        </button>
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}
