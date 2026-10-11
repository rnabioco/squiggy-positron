/**
 * Ambient declaration for side-effect CSS imports (`import './foo.css'`).
 *
 * The webview bundle loads these through webpack's style-loader/css-loader.
 * TypeScript 6 enables `noUncheckedSideEffectImports` by default, so the
 * module must be declared for `tsc` to resolve it.
 */
declare module '*.css';
