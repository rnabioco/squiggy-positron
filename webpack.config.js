const path = require('path');

/**
 * Webpack configuration for Squiggy extension
 *
 * Creates two bundles:
 * 1. Extension bundle - Node.js environment (VSCode extension host)
 * 2. Webview bundle - Browser environment (React UI for reads panel)
 *
 * Note: Demo session data is bundled with the Python package, not the extension.
 */

/** @type {import('webpack').Configuration} */
const extensionConfig = {
    name: 'extension',
    target: 'node',
    mode: 'none',
    entry: './src/extension.ts',
    output: {
        path: path.resolve(__dirname, 'build'),
        filename: 'extension.js',
        libraryTarget: 'commonjs2',
        devtoolModuleFilenameTemplate: '../[resource-path]',
    },
    devtool: 'nosources-source-map',
    externals: {
        vscode: 'commonjs vscode', // Don't bundle vscode module
        positron: 'commonjs positron', // Don't bundle positron module
    },
    resolve: {
        extensions: ['.ts', '.js'],
    },
    module: {
        rules: [
            {
                test: /\.ts$/,
                exclude: /node_modules/,
                use: [
                    {
                        loader: 'ts-loader',
                        options: {
                            configFile: 'tsconfig.json',
                        },
                    },
                ],
            },
        ],
    },
};

/** @type {import('webpack').Configuration} */
const webviewConfig = {
    name: 'webview',
    target: 'web',
    mode: 'none',
    entry: './src/views/components/webview-entry.tsx',
    output: {
        path: path.resolve(__dirname, 'build'),
        filename: 'webview.js',
    },
    devtool: 'nosources-source-map',
    resolve: {
        extensions: ['.tsx', '.ts', '.js'],
    },
    module: {
        rules: [
            {
                test: /\.tsx?$/,
                exclude: /node_modules/,
                use: [
                    {
                        loader: 'ts-loader',
                        options: {
                            configFile: 'tsconfig.json',
                        },
                    },
                ],
            },
            {
                test: /\.css$/,
                use: ['style-loader', 'css-loader'],
            },
        ],
    },
};

module.exports = [extensionConfig, webviewConfig];
