chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "analyze") {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000);

    fetch("http://127.0.0.1:8000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message.data),
      signal: controller.signal
    })
    .then(res => {
      clearTimeout(timeoutId);
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
      console.error("서버 통신 실패:", err.message);
      sendResponse({ success: false, is_chart: false, error: err.message });
    });

    return true;
  }
});