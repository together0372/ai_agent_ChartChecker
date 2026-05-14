chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "analyze") {
    fetch("http://127.0.0.1:8000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message.data)
    })
    .then(res => res.json())
    .then(data => {
      console.log("Server response:", data);
      sendResponse(data);
    })
    .catch(err => console.error(err));
    return true; // async response
  }
});