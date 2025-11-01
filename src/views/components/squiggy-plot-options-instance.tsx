/**
 * Plot Options Instance - Webview Entry Point
 *
 * Top-level component for the Plot Options panel webview
 */

import React from 'react';
import { PlotOptionsCore } from './squiggy-plot-options-core';

export const PlotOptionsInstance: React.FC = () => {
    return <PlotOptionsCore />;
};
