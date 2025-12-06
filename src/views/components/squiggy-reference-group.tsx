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
    nameColumnWidth: _nameColumnWidth,
    detailsColumnWidth,
    onToggle,
}) => {
    const handleClick = () => {
        onToggle(item.referenceName);
    };

    return (
        <div
            className={`reference-group ${isEvenRow ? 'even-row' : 'odd-row'}`}
            onClick={handleClick}
            title={`${item.referenceName} (${item.readCount} reads)`}
        >
            {/* Name column - Reference name with chevron */}
            <div className="reference-group-column reference-group-name">
                <span className={`reference-group-chevron ${item.isExpanded ? 'expanded' : ''}`}>
                    ▶
                </span>
                <span className="reference-group-icon">📁</span>
                <span className="reference-group-label">{item.referenceName}</span>
            </div>

            {/* Column divider */}
            <div className="reference-group-divider" />

            {/* Details column - Read count */}
            <div
                className="reference-group-column reference-group-details"
                style={{ width: `${detailsColumnWidth}px` }}
            >
                <span className="reference-group-count">{item.readCount}</span>
            </div>

            {/* Actions column - empty for reference groups to maintain alignment */}
            <div className="reference-group-column reference-group-actions" />
        </div>
    );
};
