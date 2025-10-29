---
description: Launch Squiggy app with test data
---

Launch the Squiggy application with test POD5 and BAM files, then return control to the console.

Run this command in the background:
```bash
squiggy -p tests/data/yeast_trna_reads.pod5 -b tests/data/yeast_trna_mappings.bam &
```

The application will open in a new window while you retain control of the terminal.

IMPORTANT: After running this command, STOP immediately. Do not monitor output, wait for completion, or continue thinking. The `&` returns control instantly.
