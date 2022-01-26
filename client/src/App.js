import React, { useState } from "react";
import Card from "./components/Card";
import "./App.css";

const App = () => {
  const [file, setFile] = useState(null);
  const [paths, setPaths] = useState();
  const [loadingState, setLoadingState] = useState(0);

  const onChangeHandler = (event) => {
    setFile(event.target.files[0]);
    console.log(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoadingState(1)
    const uploadData = new FormData();
    uploadData.append("video", file);

    console.log(uploadData.get("cover"));

    return fetch("http://127.0.0.1:8000/api/upload/", {
      method: "POST",
      body: uploadData,
    })
      .then((response) => {
        return response.json();
      })
      .then((response) => {
        setPaths(response);
        setLoadingState(2);
      });
  };

  return (
    <div className="wrapper">
      <form className="upload-form" onSubmit={handleSubmit}>
        <div className="footer">
          <label className="csv-select">
            <input type="file" name="file" onChange={onChangeHandler} />
            Select File
          </label>
          <button className="result-upload-btn ">Upload</button>
        </div>
      </form>
      <div className="grouped">
        {/* Template for a Card (== reference image + place for assigned frame) */}
        <Card id={0} name={"Badge (front)"} paths={paths} loadingState={loadingState} />
      </div>
    </div>
  );
};

export default App;
