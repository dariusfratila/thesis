import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import styles from "./Menu.module.css";

function Menu({ isVisible }) {
  const [hoveredItem, setHoveredItem] = useState(null);
  const navigate = useNavigate();

  const handleMouseEnter = (item) => setHoveredItem(item);
  const handleMouseLeave = () => setHoveredItem(null);

  const handleNavigation = (item) => {
    switch (item) {
      case "Approach":
        navigate("/approach");
        break;
      case "FAQ":
        navigate("/faq");
        break;
      case "Demo":
        navigate("/demo");
        break;
      default:
        navigate("/");
    }
  };

  return (
    <nav
      className={`${styles.menu} ${isVisible ? styles.menuVisible : ""}`}
      aria-hidden={!isVisible}
    >
      <ul className={styles.menuList}>
        {["Approach", "FAQ", "Demo"].map((item, index) => (
          <li
            key={index}
            className={`${styles.menuItem} ${
              hoveredItem && hoveredItem !== item ? styles.inactive : ""
            }`}
            onMouseEnter={() => handleMouseEnter(item)}
            onMouseLeave={handleMouseLeave}
            onClick={() => handleNavigation(item)}
            tabIndex={0}
          >
            {item}
          </li>
        ))}
      </ul>
    </nav>
  );
}

export default Menu;
