$(document).ready(function () {
  let i = 5;
  let timer;
  let liveDataTimer = null;
  let lastValidPPGMessage = "";

  function start(modalSelector) {
    $(`${modalSelector} .modal-body`).html(`<p style="font-size: 50px">Start!</p>`);
    setTimeout(() => $(`${modalSelector} .modal-body`).html(""), 1000);
  }

  function updateNumber(modalSelector, callback) {
    if (i > 1) {
      $(`${modalSelector} .modal-body`).html(`<p style="font-size: 50px">The exercise starts in ${i} seconds</p>`);
    } else {
      $(`${modalSelector} .modal-body`).html(`<p style="font-size: 50px">The exercise starts in ${i} second</p>`);
    }

    if (i > 0) {
      i--;
      timer = setTimeout(() => updateNumber(modalSelector, callback), 1000);
    } else {
      start(modalSelector);
      setTimeout(callback, 1000);
    }
  }

  function updateNumberPPG(modalSelector, callback) {
    if (i > 1) {
      $(`${modalSelector} .modal-body`).html(`<p style="font-size: 30px">The analysis of the heart rhythm starts in ${i} seconds</p>`);
    } else {
      $(`${modalSelector} .modal-body`).html(`<p style="font-size: 30px">The analysis of the heart rhythm starts in ${i} second</p>`);
    }

    if (i > 0) {
      i--;
      timer = setTimeout(() => updateNumberPPG(modalSelector, callback), 1000);
    } else {
      start(modalSelector);
      setTimeout(callback, 1000);
    }
  }

  function monitorMovement() {
    axios.get("http://192.168.0.171:8081/data")
      .then((response) => {
        const data = response.data;
        axios.post("http://192.168.0.119:5000/predict_delta", data)
          .then((res) => {
            const result = res.data;
            let msg = "";

            if (result.final_result === 1) {
              msg = `<p style='font-size: 30px; color: green;'>Correct repetition</p>
                     <p style='font-size: 20px;'>Consecutive correct: ${result.consecutive_correct}</p>`;
            } else if (result.final_result === 0) {
              msg = `<p style='font-size: 30px; color: red;'>Incorrect repetition</p>
                     <p style='font-size: 20px;'>Consecutive wrong: ${result.consecutive_wrong}</p>`;
            } else {
              msg = `<p style='font-size: 30px; color: orange;'>Waiting...</p>`;
            }

            $("#myModal .modal-body").html(msg);
          })
          .catch((err) => {
            console.error("Flask error (delta):", err);
            $("#myModal .modal-body").html("<p style='color: red'>Flask communication error</p>");
          });

        liveDataTimer = setTimeout(monitorMovement, 20);
      })
      .catch((error) => {
        console.error("ESP32 error (delta):", error);
        $("#myModal .modal-body").html("<p style='color: red'>ESP32 connection error</p>");
      });
  }

  function monitorPPG() {
    axios.get("http://192.168.0.171:8081/data")
      .then((response) => {
        const rawPPG = response.data.ppgRaw;

        if (typeof rawPPG !== 'number' || isNaN(rawPPG)) {
          console.warn("Invalid PPG value:", rawPPG);
          $("#myModal1 .modal-body").html("<p style='color: orange'>Waiting for PPG signal...</p>");
          liveDataTimer = setTimeout(monitorPPG, 1000);
          return;
        }

        const ppgData = { ppgRaw: rawPPG };

        axios.post("http://192.168.0.119:5000/predict_ppg", ppgData)
          .then((res) => {
            const result = res.data;
            let msg = "";

            if (result.ppg_result === 1) {
              msg = `<p style='font-size: 30px; color: red;'>PPG: Atrial fibrillation detected</p>`;
              lastValidPPGMessage = msg;
            } else if (result.ppg_result === 0) {
              msg = `<p style='font-size: 30px; color: green;'>PPG: Normal rhythm</p>`;
              lastValidPPGMessage = msg;
            } else if (result.confidence === 0.0 && result.ppg_score === 0.0) {
              msg = `<p style='font-size: 30px; color: orange;'>Weak signal detected</p>`;
            } else {
              msg = lastValidPPGMessage || `<p style='font-size: 30px; color: orange;'>PPG: Waiting...</p>`;
            }

            $("#myModal1 .modal-body").html(msg);
            liveDataTimer = setTimeout(monitorPPG, 40);
          })
          .catch((err) => {
            console.error("Flask (PPG):", err);
            $("#myModal1 .modal-body").html("<p style='color: red'>Flask communication error (PPG)</p>");
            liveDataTimer = setTimeout(monitorPPG, 2000);
          });
      })
      .catch((err) => {
        console.error("ESP32 (PPG):", err);
        $("#myModal1 .modal-body").html("<p style='color: red'>ESP32 connection error (PPG)</p>");
        liveDataTimer = setTimeout(monitorPPG, 2000);
      });
  }

  $("#myModal, #myModal1").on("hidden.bs.modal", () => {
    clearTimeout(liveDataTimer);
    clearTimeout(timer);
    i = 5;
  });

  $(".button").on("click", (e) => {
    e.preventDefault();
    const btn = $(e.currentTarget).attr("id");

    if (btn === "button3") {
      const modal = new bootstrap.Modal($("#myModal"));
      modal.show();
      updateNumber("#myModal", monitorMovement);
    }

    if (btn === "PPG") {
      const modal = new bootstrap.Modal($("#myModal1"));
      modal.show();
      updateNumberPPG("#myModal1", monitorPPG);
    }
  });
});
