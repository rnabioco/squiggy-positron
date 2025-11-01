/**
 * ColumnResizer Component
 *
 * Draggable vertical divider for resizing columns.
 * Based on Positron's Variables panel column resizer pattern.
 */

import * as React from 'react';
import './column-resizer.css';

interface ColumnResizerProps {
    onResize: (deltaX: number) => void;
}

export const ColumnResizer: React.FC<ColumnResizerProps> = ({ onResize }) => {
    const [isDragging, setIsDragging] = React.useState(false);
    const startXRef = React.useRef<number>(0);

    const handleMouseDown = (e: React.MouseEvent) => {
        e.preventDefault();
        setIsDragging(true);
        startXRef.current = e.clientX;
    };

    React.useEffect(() => {
        if (!isDragging) {
            return;
        }

        const handleMouseMove = (e: MouseEvent) => {
            const deltaX = e.clientX - startXRef.current;
            onResize(deltaX);
            startXRef.current = e.clientX;
        };

        const handleMouseUp = () => {
            setIsDragging(false);
        };

        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);

        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
    }, [isDragging, onResize]);

    return (
        <div
            className={`column-resizer ${isDragging ? 'dragging' : ''}`}
            onMouseDown={handleMouseDown}
        />
    );
};
