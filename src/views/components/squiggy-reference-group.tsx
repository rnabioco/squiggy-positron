/**
 * ReferenceGroupComponent
 *
 * Renders a collapsible reference group header in the reads list.
 * Shows reference name, read count, and expansion chevron.
 */

import * as React from 'react';
import { ReferenceGroupProps } from '../../types/squiggy-reads-types';
import './squiggy-reference-group.css';

export const ReferenceGroupComponent: React.FC<ReferenceGroupProps> = ({
    item,
    isEvenRow,
    nameColumnWidth,
    detailsColumnWidth,
    onToggle,
    onPlotAggregate,
}) => {
    const handleClick = () => {
        onToggle(item.referenceName);
    };

    const handlePlotClick = (e: React.MouseEvent) => {
        e.stopPropagation();
        if (onPlotAggregate) {
            onPlotAggregate(item.referenceName);
        }
    };

    const formatCount = (count: number): string => {
        if (count >= 1000000) {
            return `${(count / 1000000).toFixed(1)}M`;
        } else if (count >= 1000) {
            return `${(count / 1000).toFixed(1)}K`;
        }
        return count.toString();
    };

    return (
        <div
            className={`reference-group ${isEvenRow ? 'even-row' : 'odd-row'}`}
            onClick={handleClick}
            title={`${item.referenceName} (${item.readCount} reads)`}
        >
            {/* Name column - Reference name with chevron */}
            <div
                className="reference-group-column reference-group-name"
                style={{ width: `${nameColumnWidth}px` }}
            >
                <span className={`reference-group-chevron ${item.isExpanded ? 'expanded' : ''}`}>
                    ▶
                </span>
                <span className="reference-group-icon">📁</span>
                <span className="reference-group-label">{item.referenceName}</span>
            </div>

            {/* Details column - Read count */}
            <div
                className="reference-group-column reference-group-details"
                style={{ width: `${detailsColumnWidth}px` }}
            >
                <span className="reference-group-count">{formatCount(item.readCount)}</span>
            </div>

            {/* Actions column - Aggregate plot button */}
            <div className="reference-group-column reference-group-actions">
                <button
                    className="reference-group-action-button"
                    onClick={handlePlotClick}
                    title="Plot aggregate for this reference"
                >
                    Aggregate
                </button>
            </div>
        </div>
    );
};
