import React, { useState } from "react";
import { Grid } from "react-loader-spinner";
import "./Card.css";

const Card = ({ id, name, paths, loadingState }) => {
  return (
    <div className="card-outer">
      <div className="marker">{name}</div>
      <div className="card-content">
        <img src={`${id}.jpeg`} className="part-imgs" />
        {loadingState == 0 ? (
          <div style={{ height: "90%", width: "50%" }}></div>
        ) : loadingState == 1 ? (
          <div style={{ height: "90%", width: "50%" }} className="placeholder">
            <Grid color="#003961" arialLabel="loading" />
          </div>
        ) : (
          <img
          // public GCP folder. Will be closed down after challenge.
            src={`https://storage.cloud.google.com/mtu/frames/${paths[id]}`}
            className="part-imgs"
          />
        )}
      </div>
    </div>
  );
};

export default Card;
