import React, { useState, useEffect } from "react";
import styles from "./Header.module.css";

function Header({ isMenuOpen, toggleMenu }) {
  const [isTransitioning, setIsTransitioning] = useState(false);

  useEffect(() => {
    let timeoutId;
    if (isTransitioning) {
      timeoutId = setTimeout(() => {
        toggleMenu();
        setIsTransitioning(false);
      }, 300);
    }
    return () => clearTimeout(timeoutId);
  }, [isTransitioning, toggleMenu]);  

  const handleClick = () => {
    setIsTransitioning(true);
  };

  return (
    <header className={styles.header}>
      <div className={styles.logo}>
        <h2>
          <a href="/">LIPREAD AI</a>
        </h2>
      </div>
      <div
        role="button"
        aria-expanded={isMenuOpen}
        className={`${styles.menuIcon} ${
          isTransitioning ? styles.menuIconTransition : ""
        }`}
        onClick={handleClick}
        onKeyDown={handleClick}
        tabIndex={0}
      >                                                     
        {isMenuOpen ? "CLOSE" : "MENU"}
      </div>
    </header>
  );
}

export default Header;
