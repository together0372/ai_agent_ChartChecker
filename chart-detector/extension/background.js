chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "analyze") {
    // 127.0.0.1 주소로 FastAPI 서버에 요청
    fetch("http://127.0.0.1:8000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message.data)
    })
    .then(res => {
      if (!res.ok) {
        throw new Error("서버 에러 발생: " + res.status);
      }
      return res.json();
    })
    .then(data => {
      console.log("서버 정상 응답:", data);
      sendResponse(data);
    })
    .catch(err => {
      console.error("통신 실패 (Failed to fetch):", err);
      // 에러가 났을 때 프론트엔드(content.js)가 무한 대기하지 않도록 빈 응답이라도 쏴줍니다.
      sendResponse({ success: false, is_chart: false, error: err.message });
    });
    
    return true; // 비동기 응답(sendResponse)을 위해 반드시 필요
  }
});