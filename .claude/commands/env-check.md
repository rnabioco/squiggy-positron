# description: Check development environment setup and dependencies

Verify the development environment is properly configured:

## Environment Check

1. **Check Python version and packages**
   - Run `python3.12 --version` to verify Python 3.12 is available
   - Run `python3.12 -m pip list | grep -E "(squiggy|pytest|ruff|bokeh|pod5|pysam|numpy)"` to check installed packages

2. **Check Node.js and npm packages**
   - Run `node --version` to verify Node.js version
   - Run `npm --version` to verify npm version
   - Run `ls node_modules | wc -l` to count installed packages (should be ~730+)

3. **Check build tools**
   - Run `node node_modules/typescript/bin/tsc --version` (6.x, JS API) and `node node_modules/@typescript/native/lib/tsc.js --version` (7.x, native) to verify TypeScript
   - Run `npx webpack --version` to verify webpack
   - Run `npx jest --version` to verify jest

4. **Check environment variables**
   - Run `echo $PYTHON` to verify Python path is set
   - Run `which python` to see which Python is default

5. **Quick smoke test**
   - Run `npm run compile` to verify extension can build
   - Run `python3.12 -c "import squiggy; print(squiggy.__version__)"` to verify Python package loads

Report:
- ✅ All tools installed and versions
- ⚠️ Any missing dependencies
- ❌ Any errors or configuration issues
- 💡 Recommendations for fixing issues

**This command is especially useful after SessionStart to verify the environment setup completed successfully.**
