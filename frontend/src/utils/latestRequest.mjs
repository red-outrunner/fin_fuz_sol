// Abort superseded work and also guard responses from transports that ignore abort.
export function createLatestRequest() {
    let current = null;
    return {
        begin() {
            current?.abort();
            const controller = new AbortController();
            current = controller;
            return {
                signal: controller.signal,
                isCurrent: () => current === controller && !controller.signal.aborted,
            };
        },
        cancel() {
            current?.abort();
            current = null;
        },
    };
}
