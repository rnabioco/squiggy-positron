/**
 * Tests for TSVParser
 */

import { TSVParser } from '../tsv-parser';

describe('TSVParser', () => {
    describe('parse() - valid input', () => {
        test('parses valid TSV with all columns', () => {
            const content = `sample_name\tpod5\tbam\tfasta
sample_A\tdata/A.pod5\tdata/A.bam\tref.fa
sample_B\tdata/B.pod5\tdata/B.bam\tref.fa`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(true);
            expect(result.errors).toHaveLength(0);
            expect(result.samples).toHaveLength(2);

            expect(result.samples[0]).toEqual({
                sampleName: 'sample_A',
                pod5Path: 'data/A.pod5',
                bamPath: 'data/A.bam',
                fastaPath: 'ref.fa',
                lineNumber: 2,
            });

            expect(result.samples[1]).toEqual({
                sampleName: 'sample_B',
                pod5Path: 'data/B.pod5',
                bamPath: 'data/B.bam',
                fastaPath: 'ref.fa',
                lineNumber: 3,
            });
        });

        test('parses TSV with optional columns missing (no BAM)', () => {
            const content = `sample_name\tpod5\tbam\tfasta
sample_A\tdata/A.pod5\t-\tref.fa
sample_B\tdata/B.pod5\t\tref.fa`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(true);
            expect(result.samples).toHaveLength(2);
            expect(result.samples[0].bamPath).toBeUndefined();
            expect(result.samples[1].bamPath).toBeUndefined();
            expect(result.warnings.length).toBeGreaterThan(0);
            expect(result.warnings[0]).toContain('no BAM file');
        });

        test('parses TSV with optional columns missing (no FASTA)', () => {
            const content = `sample_name\tpod5\tbam\tfasta
sample_A\tdata/A.pod5\tdata/A.bam\t-
sample_B\tdata/B.pod5\tdata/B.bam\t`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(true);
            expect(result.samples).toHaveLength(2);
            expect(result.samples[0].fastaPath).toBeUndefined();
            expect(result.samples[1].fastaPath).toBeUndefined();
            expect(result.warnings.length).toBeGreaterThan(0);
            expect(result.warnings[0]).toContain('no FASTA file');
        });

        test('parses TSV with only required columns', () => {
            const content = `sample_name\tpod5
sample_A\tdata/A.pod5
sample_B\tdata/B.pod5`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(true);
            expect(result.samples).toHaveLength(2);
            expect(result.samples[0].bamPath).toBeUndefined();
            expect(result.samples[0].fastaPath).toBeUndefined();
        });

        test('handles Windows line endings (CRLF)', () => {
            const content = `sample_name\tpod5\r\nsample_A\tdata/A.pod5\r\nsample_B\tdata/B.pod5`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(true);
            expect(result.samples).toHaveLength(2);
        });

        test('skips empty lines', () => {
            const content = `sample_name\tpod5

sample_A\tdata/A.pod5

sample_B\tdata/B.pod5

`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(true);
            expect(result.samples).toHaveLength(2);
        });

        test('skips comment lines starting with #', () => {
            const content = `# This is a comment
sample_name\tpod5
# Another comment
sample_A\tdata/A.pod5
sample_B\tdata/B.pod5`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(true);
            expect(result.samples).toHaveLength(2);
        });

        test('auto-detects comma delimiter', () => {
            const content = `sample_name,pod5,bam
sample_A,data/A.pod5,data/A.bam
sample_B,data/B.pod5,data/B.bam`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(true);
            expect(result.samples).toHaveLength(2);
            expect(result.samples[0].pod5Path).toBe('data/A.pod5');
            expect(result.samples[0].bamPath).toBe('data/A.bam');
        });

        test('handles mixed case column names', () => {
            const content = `Sample_Name\tPOD5\tBAM
sample_A\tdata/A.pod5\tdata/A.bam`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(true);
            expect(result.samples).toHaveLength(1);
        });

        test('trims whitespace from cells', () => {
            const content = `sample_name\tpod5\tbam
  sample_A  \t  data/A.pod5  \t  data/A.bam  `;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(true);
            expect(result.samples[0].sampleName).toBe('sample_A');
            expect(result.samples[0].pod5Path).toBe('data/A.pod5');
            expect(result.samples[0].bamPath).toBe('data/A.bam');
        });
    });

    describe('parse() - error cases', () => {
        test('rejects empty file', () => {
            const content = '';

            const result = TSVParser.parse(content);

            expect(result.success).toBe(false);
            expect(result.errors).toContain('TSV file is empty');
        });

        test('rejects file with only whitespace', () => {
            const content = '   \n  \n  ';

            const result = TSVParser.parse(content);

            expect(result.success).toBe(false);
            expect(result.errors).toContain('TSV file is empty');
        });

        test('rejects TSV missing sample_name column', () => {
            const content = `pod5\tbam
data/A.pod5\tdata/A.bam`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(false);
            expect(result.errors).toContain('Missing required column: sample_name');
        });

        test('rejects TSV missing pod5 column', () => {
            const content = `sample_name\tbam
sample_A\tdata/A.bam`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(false);
            expect(result.errors).toContain('Missing required column: pod5');
        });

        test('rejects rows with missing sample_name', () => {
            const content = `sample_name\tpod5
\tdata/A.pod5
sample_B\tdata/B.pod5`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(false);
            expect(result.errors).toContain('Line 2: Missing sample_name');
        });

        test('rejects rows with missing pod5 path', () => {
            const content = `sample_name\tpod5
sample_A\t
sample_B\tdata/B.pod5`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(false);
            expect(result.errors).toContain(
                "Line 2: Missing pod5 path for sample 'sample_A'"
            );
        });

        test('rejects rows with pod5 path as dash', () => {
            const content = `sample_name\tpod5
sample_A\t-
sample_B\tdata/B.pod5`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(false);
            expect(result.errors).toContain(
                "Line 2: Missing pod5 path for sample 'sample_A'"
            );
        });

        test('detects duplicate sample names', () => {
            const content = `sample_name\tpod5
sample_A\tdata/A.pod5
sample_A\tdata/A2.pod5`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(false);
            expect(result.errors).toContain("Line 3: Duplicate sample name 'sample_A'");
        });

        test('reports multiple errors', () => {
            const content = `sample_name\tpod5
\tdata/A.pod5
sample_B\t
sample_C\tdata/C.pod5
sample_C\tdata/C2.pod5`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(false);
            expect(result.errors.length).toBeGreaterThanOrEqual(3);
        });
    });

    describe('parse() - warnings', () => {
        test('warns when BAM is missing', () => {
            const content = `sample_name\tpod5\tbam
sample_A\tdata/A.pod5\t-`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(true);
            expect(result.warnings.length).toBeGreaterThan(0);
            expect(result.warnings[0]).toContain('no BAM file');
            expect(result.warnings[0]).toContain('alignment features unavailable');
        });

        test('warns when FASTA is missing', () => {
            const content = `sample_name\tpod5\tfasta
sample_A\tdata/A.pod5\t-`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(true);
            expect(result.warnings.length).toBeGreaterThan(0);
            expect(result.warnings[0]).toContain('no FASTA file');
            expect(result.warnings[0]).toContain('reference sequence unavailable');
        });

        test('warns for multiple missing optional files', () => {
            const content = `sample_name\tpod5\tbam\tfasta
sample_A\tdata/A.pod5\t-\t-`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(true);
            expect(result.warnings.length).toBe(2); // One for BAM, one for FASTA
        });
    });

    describe('parse() - edge cases', () => {
        test('handles single sample', () => {
            const content = `sample_name\tpod5
sample_A\tdata/A.pod5`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(true);
            expect(result.samples).toHaveLength(1);
        });

        test('handles many samples (24)', () => {
            const rows = ['sample_name\tpod5'];
            for (let i = 1; i <= 24; i++) {
                rows.push(`sample_${i}\tdata/sample_${i}.pod5`);
            }
            const content = rows.join('\n');

            const result = TSVParser.parse(content);

            expect(result.success).toBe(true);
            expect(result.samples).toHaveLength(24);
        });

        test('preserves line numbers correctly', () => {
            const content = `# Comment line
sample_name\tpod5

sample_A\tdata/A.pod5

sample_B\tdata/B.pod5`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(true);
            // Line numbers should account for empty lines and comments
            expect(result.samples[0].lineNumber).toBeDefined();
            expect(result.samples[1].lineNumber).toBeDefined();
        });

        test('handles file paths with spaces', () => {
            const content = `sample_name\tpod5\tbam
sample_A\tdata/my file.pod5\tdata/my file.bam`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(true);
            expect(result.samples[0].pod5Path).toBe('data/my file.pod5');
            expect(result.samples[0].bamPath).toBe('data/my file.bam');
        });

        test('handles absolute paths', () => {
            const content = `sample_name\tpod5
sample_A\t/absolute/path/to/A.pod5`;

            const result = TSVParser.parse(content);

            expect(result.success).toBe(true);
            expect(result.samples[0].pod5Path).toBe('/absolute/path/to/A.pod5');
        });
    });
});
