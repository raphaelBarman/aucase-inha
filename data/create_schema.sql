CREATE TABLE `actor` (
  `actor_id` int(11) NOT NULL,
  `first_name` varchar(65) CHARACTER SET utf8 DEFAULT NULL,
  `last_name` varchar(45) CHARACTER SET utf8 NOT NULL,
  `role` varchar(45) CHARACTER SET utf8 DEFAULT NULL,
  PRIMARY KEY (`actor_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `sale` (
  `sale_id` int(11) NOT NULL,
  `date` datetime NOT NULL,
  `cote_inha` varchar(45) NOT NULL,
  `url_inha` varchar(75) NOT NULL,
  PRIMARY KEY (`sale_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `actor_sale` (
  `actor_id` int(11) NOT NULL,
  `sale_id` int(11) NOT NULL,
  PRIMARY KEY (`actor_id`,`sale_id`),
  KEY `sale_id_idx` (`sale_id`),
  CONSTRAINT `actor_id` FOREIGN KEY (`actor_id`) REFERENCES `actor` (`actor_id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `sale_id` FOREIGN KEY (`sale_id`) REFERENCES `sale` (`sale_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `section` (
  `sale_id` int(11) NOT NULL,
  `page` double NOT NULL,
  `entity` int(11) NOT NULL,
  `parent_section_sale` int(11) DEFAULT NULL,
  `parent_section_page` double DEFAULT NULL,
  `parent_section_entity` int(11) DEFAULT NULL,
  `class` varchar(25) COLLATE utf8mb4_unicode_ci NOT NULL,
  `text` mediumtext COLLATE utf8mb4_unicode_ci NOT NULL,
  `bbox` varchar(45) COLLATE utf8mb4_unicode_ci NOT NULL,
  `inha_url` varchar(175) COLLATE utf8mb4_unicode_ci NOT NULL,
  `iiif_url` varchar(175) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  PRIMARY KEY (`sale_id`,`page`,`entity`),
  KEY `parent section_idx` (`parent_section_page`,`parent_section_sale`,`parent_section_entity`),
  KEY `parent_section` (`parent_section_sale`,`parent_section_page`,`parent_section_entity`),
  FULLTEXT KEY `text` (`text`),
  CONSTRAINT `parent_section` FOREIGN KEY (`parent_section_sale`, `parent_section_page`, `parent_section_entity`) REFERENCES `section` (`sale_id`, `page`, `entity`),
  CONSTRAINT `sale id section` FOREIGN KEY (`sale_id`) REFERENCES `sale` (`sale_id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `object` (
  `sale_id` int(11) NOT NULL,
  `page` double NOT NULL,
  `entity` int(11) NOT NULL,
  `parent_section_sale` int(11) DEFAULT NULL,
  `parent_section_page` double DEFAULT NULL,
  `parent_section_entity` int(11) DEFAULT NULL,
  `num_ref` varchar(45) CHARACTER SET utf8 DEFAULT NULL,
  `text` mediumtext CHARACTER SET utf8 NOT NULL,
  `bbox` varchar(45) CHARACTER SET utf8 NOT NULL,
  `inha_url` varchar(175) CHARACTER SET utf8 NOT NULL,
  `iiif_url` varchar(175) CHARACTER SET utf8 DEFAULT NULL,
  PRIMARY KEY (`sale_id`,`page`,`entity`),
  KEY `parent section object_idx` (`parent_section_sale`,`parent_section_page`,`parent_section_entity`),
  FULLTEXT KEY `text` (`text`),
  CONSTRAINT `parent section object` FOREIGN KEY (`parent_section_sale`, `parent_section_page`, `parent_section_entity`) REFERENCES `section` (`sale_id`, `page`, `entity`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `sale id object` FOREIGN KEY (`sale_id`) REFERENCES `sale` (`sale_id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
