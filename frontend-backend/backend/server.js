const express = require("express");
const cors = require("cors");
const { spawn } = require("child_process");

const app = express();

app.use(express.json());

app.use(
    cors({
        origin: "http://localhost:5173",
        methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        credentials: true,
    })
);

app.get("/", (req, res) => {
    console.log("Hello");
    res.status(200).send({ response: "You are good" });
});

const executePython = async (script, args) => {
    const arguments = args.map((arg) => arg.toString());
    const py = spawn("python", [script, ...arguments]);

    let stdOutData = "";
    let stdErrData = "";


    const result = await new Promise((resolve, reject) => {
        // let output;

        py.stdout.on("data", (data) => {
            // output = JSON.parse(data);
            stdOutData += data.toString();
        });

        py.stderr.on("data", (data) => {
            stdErrData += data.toString();
            // reject(`Error occured in ${script}`);
        });

        py.on("exit", (code) => {
            if(code !== 0) {
                console.error(`[python] Error occured: ${stdErrData}`);
                reject(new Error(`Program has exited with code ${code}`))
            }

            try {
                const output = JSON.parse(stdOutData);
                const result = JSON.parse(output.generated_text)
                console.log(`[python] Child process executed with code ${code}`);
                resolve(result);
            } catch (err) {
                console.error(`Failed to parse JSON: ${stdOutData}`);
                reject(new Error(`Invalid JSON from Python: ${err.message}`))
            }
        });
    });

    return result;
};

app.post("/api/", async (req, res) => {
    const { transcript } = req.body;
    try {
        const result = await executePython("../../ml-service/interact_ai_model.py", [
            transcript,
        ]);
        res.json({ message: "Got it!", result });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.listen(9000, () => {
    console.log("Server RUNNING");
});
