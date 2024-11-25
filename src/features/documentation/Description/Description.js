import React, { forwardRef } from "react";
import styles from "./Description.module.css";
import { useNavigate } from "react-router-dom";

const Description = forwardRef((props, ref) => {
  const navigate = useNavigate();

  const HandleButtonClick = () => {
    navigate("/demo");
  };

  return (
    <div className={styles.description}>
      {/* <h1 className={styles.title}>Revolutionize Communication</h1> */}
      <p className={styles.text}>
        Empowering Global Connectivity Through Advanced Lipreading Technology
      </p>
      <button className={styles.button} onClick={HandleButtonClick}>
        DEMO TRY NOW
      </button>
    </div>
  );
});

export default Description;
