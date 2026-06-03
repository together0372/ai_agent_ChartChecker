// Service Worker 강제 유지 (Manifest V3 장시간 요청 대비)
function keepAlive() {
  chrome.runtime.getPlatformInfo(() => {});
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "analyze") {
    const controller = new AbortController();

    // 멀티에이전트 분석 최대 5분 대기
    const timeoutId = setTimeout(() => controller.abort(), 300000);

    // Service Worker 가 중간에 종료되지 않도록 10초마다 keepAlive
    const keepAliveInterval = setInterval(keepAlive, 10000);

    fetch("http://127.0.0.1:8000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message.data),
      signal: controller.signal
    })
    .then(res => {
      clearTimeout(timeoutId);
      clearInterval(keepAliveInterval);
      if (!res.ok) {
        throw new Error("서버 에러: " + res.status);
      }
      return res.json();
    })
    .then(data => {
      console.log("서버 응답:", data);
      sendResponse(data);
    })
    .catch(err => {
      clearTimeout(timeoutId);
      clearInterval(keepAliveInterval);
      console.error("서버 통신 실패:", err.message);
      sendResponse({ success: false, is_chart: false, error: err.message });
    });

    return true;
  }
});