import React, { useState, useEffect } from "react";
import styles from "./ScrollToExplore.module.css";

function ScrollToExplore({ onClick }) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(true);
    }, 1000);

    const onScroll = () => {
      setIsVisible(false);
    };

    window.addEventListener("scroll", onScroll);

    return () => {
      clearTimeout(timer);
      window.removeEventListener("scroll", onScroll);
    };
  }, []);

  return (
    <div
      className={`${styles.scrollToExplore} ${isVisible ? styles.visible : ""}`}
      onClick={onClick}
      aria-label="Scroll down to explore more"
    >
      <p>SCROLL TO EXPLORE</p>
      <div className={styles.line}></div>
    </div>
  );
}

export default ScrollToExplore;
