const downloadedUrls = new Set();

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {

    if (msg.type === "DOWNLOAD_IMAGE") {

        const imageUrl = msg.url;

        if (!imageUrl) {
            return;
        }

        if (downloadedUrls.has(imageUrl)) {
            console.log("이미 다운로드됨:", imageUrl);
            return;
        }

        downloadedUrls.add(imageUrl);

        const cleanName = createFilename(imageUrl);

        chrome.downloads.download({
            url: imageUrl,
            filename: `AutoImageSaver/${cleanName}`,
            conflictAction: "uniquify",
            saveAs: false
        }, (downloadId) => {

            if (chrome.runtime.lastError) {
                console.error(chrome.runtime.lastError.message);
                return;
            }

            console.log("다운로드 완료:", downloadId);
        });
    }
});

function createFilename(url) {

    const timestamp = Date.now();

    try {
        const parsed = new URL(url);

        const pathname = parsed.pathname;
        const last = pathname.split("/").pop();

        if (last && last.includes(".")) {
            return `${timestamp}_${last}`;
        }

    } catch {}

    return `${timestamp}.jpg`;
}