package main;

import singleton.Database;

public class Main {

	public static void main(String[] args) {
		Database agus = Database.getConnection();
		Database siti = Database.getConnection();

	}

}


