/* Successful sections are cached; failed sections can be retried independently. */
(function (root) {
    function createRetryableSections() {
        let generation = 0;
        const completed = new Set();
        const pending = new Map();
        return {
            reset() { generation++; completed.clear(); pending.clear(); },
            async run(key, loader) {
                if (completed.has(key)) return true;
                if (pending.has(key)) return pending.get(key);
                const started = generation;
                const isCurrent = () => started === generation;
                const task = Promise.resolve().then(() => isCurrent() ? loader(isCurrent) : false).then(result => {
                    const ok = result !== false && isCurrent();
                    if (ok) completed.add(key);
                    return ok;
                }).catch(() => false).finally(() => {
                    if (isCurrent()) pending.delete(key);
                });
                pending.set(key, task);
                return task;
            }
        };
    }
    root.createRetryableSections = createRetryableSections;
    if (typeof module !== 'undefined') module.exports = { createRetryableSections };
})(globalThis);
