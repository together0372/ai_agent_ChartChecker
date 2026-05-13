document.getElementById("clearBtn").addEventListener("click", () => {

    chrome.runtime.reload();

    alert("초기화 완료");
});