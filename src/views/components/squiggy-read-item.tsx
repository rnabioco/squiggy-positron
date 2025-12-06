/**
 * ReadItemComponent
 *
 * Renders an individual read row in the reads list.
 * Shows read ID, optional genomic position, and action buttons.
 */

import * as React from 'react';
import { ReadItemProps, CONSTANTS } from '../../types/squiggy-reads-types';
import './squiggy-read-item.css';

export const ReadItemComponent: React.FC<ReadItemProps> = ({
    item,
    isSelected,
    isFocused,
    isEvenRow,
    nameColumnWidth: _nameColumnWidth,
    detailsColumnWidth,
    onPlotRead,
    onClick,
}) => {
    const handleClick = (e: React.MouseEvent) => {
        onClick(item.readId, e.ctrlKey || e.metaKey);
    };

    const handlePlotClick = (e: React.MouseEvent) => {
        e.stopPropagation();
        onPlotRead(item.readId);
    };

    const indentPx = item.indentLevel * CONSTANTS.INDENT_SIZE;

    return (
        <div
            className={`read-item ${isSelected ? 'selected' : ''} ${isFocused ? 'focused' : ''} ${isEvenRow ? 'even-row' : 'odd-row'}`}
            onClick={handleClick}
            title={item.readId}
        >
            {/* Name column - Read ID */}
            <div
                className="read-item-column read-item-name"
                style={{ paddingLeft: `${8 + indentPx}px` }}
            >
                <span className="read-item-icon">📊</span>
                <span className="read-item-id">{item.readId}</span>
            </div>

            {/* Column divider */}
            <div className="read-item-divider" />

            {/* Details column - Genomic position or reference */}
            <div
                className="read-item-column read-item-details"
                style={{ width: `${detailsColumnWidth}px` }}
            >
                {item.genomicPosition && (
                    <span className="read-item-position">{item.genomicPosition}</span>
                )}
                {item.referenceName && !item.genomicPosition && (
                    <span className="read-item-reference">{item.referenceName}</span>
                )}
                {item.quality !== undefined && (
                    <span className="read-item-quality">Q{item.quality}</span>
                )}
            </div>

            {/* Actions column */}
            <div className="read-item-column read-item-actions">
                <button
                    className="read-item-action-button"
                    onClick={handlePlotClick}
                    title="Plot this read"
                >
                    Plot
                </button>
            </div>
        </div>
    );
};
